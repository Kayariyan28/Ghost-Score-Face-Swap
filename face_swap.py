"""
Core face swap pipeline.

Two modes only:

  * ``hybrid`` (default)
    1. inswapper_128 swaps the target's face to the source identity, giving
       a pose- and expression-adapted base — handles arbitrary head pose.
    2. GFPGAN restores face structure at 2× canvas for HD output.
    3. The ORIGINAL source face image is then affine-warped onto the swapped
       face's 5-point landmarks at full source resolution (Lanczos) and the
       source pixels are alpha-composited over the swap, masked by the
       BiSeNet-detected SKIN region of the target. This is the Photoshop-
       style transplant step — it carries the source image's actual skin
       texture, pore detail, eye glints, and lip color directly into the
       output. No piecewise / Delaunay warp (which was causing black-patch
       artefacts); the warp uses a single affine which is well-defined
       everywhere.
    4. Hair / glasses / hats / earrings / scarves from the target are
       composited ON TOP of the result using BiSeNet's occluder mask so
       complex hairstyles (bangs, side-swept hair) stay intact.
    5. Optional unsharp-mask polish.

  * ``ai``
    inswapper_128 + GFPGAN, no source-pixel graft. Faster but the face is
    fully synthesized.
"""

import gc
import os
import sys

import cv2
import numpy as np

# basicsr (a GFPGAN dep) imports torchvision.transforms.functional_tensor which
# was removed in torchvision >= 0.17. Alias the new module so the import succeeds.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    import torchvision.transforms.functional as _tv_functional
    sys.modules['torchvision.transforms.functional_tensor'] = _tv_functional

import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer

from quality_metrics import compute_metrics, QualityScores

import torch
from facexlib.parsing import init_parsing_model

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False

# BiSeNet ParseNet class indices (CelebAMask-HQ schema).
# 0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g,
# 7 l_ear, 8 r_ear, 9 ear_r, 10 nose, 11 mouth, 12 u_lip, 13 l_lip,
# 14 neck, 15 neck_l, 16 cloth, 17 hair, 18 hat.
FACE_SURFACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)
OCCLUDER_CLASSES = (17, 18, 6, 9, 15, 16)


class FaceSwapper:
    def __init__(self, models_dir: str = './models'):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        # Apple Silicon acceleration: CoreMLExecutionProvider works for some
        # InsightFace models but trips a shape-inference assert on
        # 1k3d68.onnx (3D landmarks) in onnxruntime 1.17. To keep the entire
        # InsightFace stack stable we run it on CPU (native arm64 wheels).
        # The inswapper is also kept on CPU — it has custom ops CoreML can't
        # always handle. PyTorch / GFPGAN can still target MPS independently.
        cpu_only = ['CPUExecutionProvider']
        available_providers = self._pick_providers(prefer_coreml=True)
        coreml_active = any(
            (p[0] if isinstance(p, tuple) else p) == 'CoreMLExecutionProvider'
            for p in available_providers
        )
        print(f'[face_swap] onnxruntime CoreML available: {coreml_active} '
              f'(using CPU for stability)')

        self.face_app = FaceAnalysis(
            name='buffalo_l',
            root=self.models_dir,
            providers=cpu_only,
        )
        self.face_app.prepare(ctx_id=0, det_size=(1024, 1024))

        swapper_path = os.path.join(self.models_dir, 'inswapper_128.onnx')
        if not os.path.exists(swapper_path):
            raise FileNotFoundError(
                f'Missing model: {swapper_path}. Run `python download_models.py` first.'
            )
        self.swapper = insightface.model_zoo.get_model(swapper_path, providers=cpu_only)

        gfpgan_path = os.path.join(self.models_dir, 'GFPGANv1.4.pth')
        if not os.path.exists(gfpgan_path):
            raise FileNotFoundError(
                f'Missing model: {gfpgan_path}. Run `python download_models.py` first.'
            )
        # GFPGAN device: prefer MPS on Apple Silicon for ~2-3× speedup over
        # CPU. Falls back to CPU if MPS fails to load the model.
        gfpgan_device = None  # let GFPGANer pick (CUDA→CPU usually)
        try:
            import torch as _torch
            if _torch.backends.mps.is_available():
                gfpgan_device = _torch.device('mps')
                print('[face_swap] GFPGAN using MPS (Apple Silicon GPU)')
        except Exception:
            pass
        try:
            self.enhancer = GFPGANer(
                model_path=gfpgan_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=gfpgan_device,
            )
        except Exception as e:
            print(f'[face_swap] GFPGAN MPS failed ({e}); falling back to CPU')
            self.enhancer = GFPGANer(
                model_path=gfpgan_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
            )

        # BiSeNet face parser on CPU.
        try:
            self.face_parser = init_parsing_model(
                model_name='parsenet', half=False, device='cpu',
            )
            self.face_parser.eval()
            self._parser_available = True
            print('[face_swap] BiSeNet face parser loaded (CPU).')
        except Exception as e:
            print(f'[face_swap] face parser unavailable: {e}')
            self.face_parser = None
            self._parser_available = False

        # MediaPipe FaceMesh — 478 dense facial landmarks. The XNNPACK
        # delegate accumulates state under repeated invocation; we recreate
        # the detector every MEDIAPIPE_RESET_EVERY swaps to keep it healthy.
        # Pro mode runs MediaPipe multiple times per request so we keep the
        # threshold tight.
        self._mp_call_count = 0
        self.MEDIAPIPE_RESET_EVERY = 5
        if _MP_AVAILABLE:
            self.face_mesh = self._build_face_mesh()
            print('[face_swap] MediaPipe FaceMesh loaded (478 dense landmarks).')
        else:
            self.face_mesh = None

        # Per-process counter so we can log session usage.
        self._swap_count = 0

    @staticmethod
    def _build_face_mesh():
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def cleanup(self):
        """Release per-swap state — call after every swap so PyTorch (MPS)
        memory, GFPGAN's face_helper buffers, and TF Lite's tensor arena are
        all reset before the next request. Without this the process bloats
        steadily and the inference graph stalls after ~4 swaps on Apple
        Silicon."""
        try:
            # GFPGAN's face_helper holds image arrays from the last call.
            if hasattr(self.enhancer, 'face_helper'):
                self.enhancer.face_helper.clean_all()
        except Exception:
            pass
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except Exception:
            pass
        # Rebuild MediaPipe FaceMesh periodically.
        self._mp_call_count += 1
        if (self.face_mesh is not None
                and self._mp_call_count >= self.MEDIAPIPE_RESET_EVERY):
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.face_mesh = self._build_face_mesh()
            self._mp_call_count = 0
            print('[face_swap] MediaPipe FaceMesh reset (cleared TF Lite state)')
        gc.collect()

    def _dense_landmarks(self, img, face_bbox=None):
        """Return MediaPipe FaceMesh 478 landmarks as Nx2 float32 array, or
        None if detection fails. Selects the face that overlaps face_bbox if
        multiple faces are detected."""
        if self.face_mesh is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        h, w = img.shape[:2]
        best, best_score = None, -1
        for lmks in results.multi_face_landmarks:
            pts = np.array(
                [[lm.x * w, lm.y * h] for lm in lmks.landmark],
                dtype=np.float32,
            )
            if face_bbox is None:
                score = pts.shape[0]
            else:
                x1, y1, x2, y2 = face_bbox
                inside = ((pts[:, 0] >= x1) & (pts[:, 0] <= x2)
                          & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)).sum()
                score = inside
            if score > best_score:
                best_score = score
                best = pts
        return best

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _pick_providers(prefer_coreml: bool = True):
        """Pick ONNX Runtime providers, preferring CoreML on Apple Silicon
        (which uses the Apple Neural Engine + GPU) and CUDA on NVIDIA. Falls
        back to CPU if neither is available."""
        import onnxruntime as ort
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if prefer_coreml and 'CoreMLExecutionProvider' in available:
            # The "CPUAndNeuralEngine" compute unit uses ANE + CPU; works on
            # M1/M2/M3 with macOS >= 12. Allows ANE acceleration for static
            # ONNX models like SCRFD detection and ArcFace embeddings.
            return [
                ('CoreMLExecutionProvider', {
                    'MLComputeUnits': 'CPUAndNeuralEngine',
                    'ModelFormat': 'NeuralNetwork',
                }),
                'CPUExecutionProvider',
            ]
        return ['CPUExecutionProvider']

    def _pick_source_face(self, img):
        faces = self.face_app.get(img)
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: float(getattr(f, 'det_score', 1.0)) * 1000
                          + (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    def _target_faces(self, img, swap_all: bool):
        faces = self.face_app.get(img)
        if not faces:
            return []
        if swap_all:
            return faces
        return [max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))]

    def _set_enhancer_scale(self, factor: int):
        self.enhancer.upscale = factor
        if hasattr(self.enhancer, 'face_helper'):
            self.enhancer.face_helper.upscale_factor = factor

    @staticmethod
    def _unsharp_mask(img: np.ndarray, strength: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0.0)

    @staticmethod
    def _color_transfer_lab(source, target, mask):
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        m = mask > 128
        if not m.any():
            return source
        out = src_lab.copy()
        for c in range(3):
            sm = src_lab[..., c][m].mean()
            ss = src_lab[..., c][m].std() + 1e-6
            tm = tgt_lab[..., c][m].mean()
            ts = tgt_lab[..., c][m].std() + 1e-6
            out[..., c] = (src_lab[..., c] - sm) * (ts / ss) + tm
        return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ---- BiSeNet face parsing ---------------------------------------------

    def _parse_face_classes(self, img: np.ndarray) -> np.ndarray:
        """Run BiSeNet face parsing on a face crop. Returns class map at same
        resolution as `img`. None if parser unavailable."""
        if not self._parser_available:
            return None
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        t = t.unsqueeze(0)
        with torch.no_grad():
            out = self.face_parser(t)[0]
        cls = out.argmax(1)[0].byte().cpu().numpy()
        return cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)

    def _classes_around_face(self, img, face_bbox):
        """Parse a generous crop around the face_bbox and return the class
        map at full-image coordinates, plus the crop rectangle."""
        if not self._parser_available:
            return None, None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face_bbox.astype(np.int32)
        fw, fh = x2 - x1, y2 - y1
        pad_x = int(fw * 1.0)
        pad_y_top = int(fh * 1.5)
        pad_y_bot = int(fh * 0.6)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y_top)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y_bot)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None
        crop = img[cy1:cy2, cx1:cx2]
        try:
            cls = self._parse_face_classes(crop)
        except Exception:
            return None, None
        if cls is None:
            return None, None
        full = np.zeros((h, w), dtype=np.uint8)
        full[cy1:cy2, cx1:cx2] = cls
        return full, (cx1, cy1, cx2, cy2)

    def _skin_mask(self, img, face):
        """Pixel-precise mask of pixels classified as face surface."""
        cls, _ = self._classes_around_face(img, face.bbox)
        if cls is None:
            return None
        m = np.isin(cls, FACE_SURFACE_CLASSES).astype(np.uint8) * 255
        return m if m.any() else None

    def _occluder_mask(self, img, face):
        """Pixel-precise mask of pixels that occlude the face plane (hair,
        glasses, hat, scarf, earrings)."""
        cls, (cx1, cy1, cx2, cy2) = self._classes_around_face(img, face.bbox) or (None, None)
        if cls is None:
            return None
        m = np.isin(cls, OCCLUDER_CLASSES).astype(np.uint8) * 255
        if m.sum() == 0:
            return m
        fw = face.bbox[2] - face.bbox[0]
        dilate_k = max(3, int(fw * 0.012)) | 1
        m = cv2.dilate(m, np.ones((dilate_k, dilate_k), np.uint8))
        sigma = max(2.0, fw * 0.015)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma)
        return m

    # ---- public API -------------------------------------------------------

    def swap_with_metrics(self, source_img, target_img, **kwargs):
        """Run the swap pipeline and additionally return quantitative quality
        metrics (ArcFace cosine identity, SSIM, ΔE00, landmark RMSE).

        If the ArcFace cosine identity is below 0.5 (drift / partial), the
        pipeline auto-retries once with ``detail`` pushed to 1.3 and
        ``preserve_source_tone=True`` for maximum source fidelity, then
        returns whichever attempt scored higher.
        """
        import time
        t0 = time.perf_counter()
        try:
            result = self.swap(source_img, target_img, **kwargs)
        finally:
            # Always clean up — even if swap raised — so the next request
            # starts on a clean slate.
            self.cleanup()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        target_faces = self._target_faces(target_img, kwargs.get('swap_all_targets', False))
        source_face = self._pick_source_face(source_img)
        result_faces = self._target_faces(result, False) if target_faces else []
        tf_result = result_faces[0] if result_faces else None
        scores = compute_metrics(
            self.face_app,
            source_img, result, tf_result,
            dense_landmarks_fn=self._dense_landmarks,
            source_face=source_face,
            elapsed_ms=elapsed_ms,
        )

        # Auto-retry once if identity drifted, using stronger source-pixel
        # graft. Skip retry for AI mode where the result is GFPGAN-rebuilt.
        retried = False
        if (not np.isnan(scores.arcface_cosine)
                and scores.arcface_cosine < 0.50
                and kwargs.get('mode', 'photoshop') != 'ai'):
            retry_kwargs = dict(kwargs)
            retry_kwargs['detail'] = 1.3
            retry_kwargs['preserve_source_tone'] = True
            t1 = time.perf_counter()
            retry_result = self.swap(source_img, target_img, **retry_kwargs)
            retry_elapsed_ms = int((time.perf_counter() - t1) * 1000)
            r_faces = self._target_faces(retry_result, False)
            retry_scores = compute_metrics(
                self.face_app,
                source_img, retry_result, r_faces[0] if r_faces else None,
                dense_landmarks_fn=self._dense_landmarks,
                source_face=source_face,
                elapsed_ms=elapsed_ms + retry_elapsed_ms,
            )
            if (not np.isnan(retry_scores.arcface_cosine)
                    and retry_scores.arcface_cosine > scores.arcface_cosine):
                result, scores = retry_result, retry_scores
                retried = True
            self.cleanup()

        self._swap_count += 1
        return result, scores, retried

    def swap(self, source_img, target_img,
             mode='photoshop',
             enhance=True, swap_all_targets=False,
             fidelity=0.75, hd=True,
             sharpen=0.35, detail=0.9,
             preserve_hair=True,
             preserve_source_tone=False):
        if mode == 'ai':
            result = self._ai_swap(
                source_img, target_img,
                enhance=enhance, swap_all_targets=swap_all_targets,
                fidelity=fidelity, hd=hd, sharpen=sharpen,
            )
        else:
            # photoshop (default): inswapper base + Laplacian-frequency graft
            # of the source's actual skin texture + Poisson clone + hair
            # composite (the FaceFusion HD / Roop-Unleashed HD pipeline).
            result = self._photoshop_swap(
                source_img, target_img,
                enhance=enhance, swap_all_targets=swap_all_targets,
                hd=hd, sharpen=sharpen, detail=detail,
                preserve_source_tone=preserve_source_tone,
            )

        # Hair-aware composite — last step in z-order so the target's hair /
        # glasses / accessories sit on top of every face-modifying operation.
        if preserve_hair and self._parser_available:
            target_faces = self._target_faces(target_img, swap_all_targets)
            if target_faces:
                if result.shape[:2] != target_img.shape[:2]:
                    target_resized = cv2.resize(
                        target_img, (result.shape[1], result.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                else:
                    target_resized = target_img
                for tf in target_faces:
                    occ = self._occluder_mask(target_img, tf)
                    if occ is None or occ.max() == 0:
                        continue
                    if occ.shape != result.shape[:2]:
                        occ = cv2.resize(
                            occ, (result.shape[1], result.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    alpha = occ.astype(np.float32) / 255.0
                    alpha = alpha[..., None]
                    result = (result.astype(np.float32) * (1 - alpha)
                              + target_resized.astype(np.float32) * alpha)
                    result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    # ---- AI mode ----------------------------------------------------------

    def _ai_swap(self, source_img, target_img,
                 enhance, swap_all_targets, fidelity, hd, sharpen):
        source_face = self._pick_source_face(source_img)
        if source_face is None:
            raise ValueError('No face detected in the base (source) image.')
        target_faces = self._target_faces(target_img, swap_all_targets)
        if not target_faces:
            raise ValueError('No face detected in the target (body) image.')

        swapped = target_img.copy()
        for tf in target_faces:
            swapped = self.swapper.get(swapped, tf, source_face, paste_back=True)

        if enhance:
            self._set_enhancer_scale(2 if hd else 1)
            _, _, restored = self.enhancer.enhance(
                swapped, has_aligned=False,
                only_center_face=not swap_all_targets,
                paste_back=True, weight=0.5,
            )
            if swapped.shape != restored.shape:
                swapped = cv2.resize(
                    swapped, (restored.shape[1], restored.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            fidelity = float(np.clip(fidelity, 0.0, 1.0))
            if fidelity <= 0.0:
                result = restored
            elif fidelity >= 1.0:
                result = swapped
            else:
                result = cv2.addWeighted(swapped, fidelity, restored, 1.0 - fidelity, 0.0)
        else:
            result = swapped

        sharpen = float(np.clip(sharpen, 0.0, 1.5))
        if sharpen > 0.0:
            result = self._unsharp_mask(result, sharpen)
        return result

    # ---- Photoshop mode (FaceFusion-HD / Roop-Unleashed-HD pipeline) -----

    def _photoshop_swap(self, source_img, target_img,
                        enhance, swap_all_targets, hd, sharpen, detail,
                        preserve_source_tone):
        source_face = self._pick_source_face(source_img)
        if source_face is None:
            raise ValueError('No face detected in the base (source) image.')
        target_faces = self._target_faces(target_img, swap_all_targets)
        if not target_faces:
            raise ValueError('No face detected in the target (body) image.')

        # Step 1 — inswapper produces a pose / expression-adapted base.
        base = target_img.copy()
        for tf in target_faces:
            base = self.swapper.get(base, tf, source_face, paste_back=True)

        # Step 2 — GFPGAN restoration (and 2× HD upscale when hd=True).
        # The base provides the LOW-frequency structure (pose, lighting,
        # overall face shape) that the multi-scale Laplacian graft in
        # Step 3 will ADD high-frequency source detail onto.
        if enhance:
            self._set_enhancer_scale(2 if hd else 1)
            _, _, base = self.enhancer.enhance(
                base, has_aligned=False,
                only_center_face=not swap_all_targets,
                paste_back=True, weight=0.4,
            )

        # Step 3 — per target face, graft source pixels.
        rescaled_faces = self._target_faces(base, swap_all_targets)
        if not rescaled_faces:
            sharpen = float(np.clip(sharpen, 0.0, 1.5))
            return self._unsharp_mask(base, sharpen) if sharpen > 0 else base

        for tf_rescaled in rescaled_faces:
            base = self._graft_source_pixels(
                base, source_img, source_face, tf_rescaled,
                detail=detail, preserve_source_tone=preserve_source_tone,
            )

        sharpen = float(np.clip(sharpen, 0.0, 1.5))
        if sharpen > 0.0:
            base = self._unsharp_mask(base, sharpen)
        return base

    @staticmethod
    def _multi_band_blend(base, detail_source, mask, scales=(1.2, 3.0, 6.0),
                          weights=(0.55, 0.35, 0.20), detail_strength=1.0):
        """Multi-scale Laplacian-pyramid graft.

        Adds the high-frequency content of `detail_source` onto `base`
        inside `mask`, at multiple spatial scales. This is the "frequency
        domain blend" step from the FaceFusion HD / Roop-Unleashed HD
        pipeline: the result has

            low_freq  ← inswapper base (correct pose / lighting)
            high_freq ← warped source  (real skin pores, eye glints, hair
                                        strands at native source resolution)

        No alpha blending of two faces, so no "double image" — we are
        ADDING source's high-freq detail onto base, not REPLACING base.
        """
        base_f = base.astype(np.float32)
        out = base_f.copy()
        # Soft feather of the mask so the graft fades smoothly at the edge.
        feather = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0).astype(np.float32) / 255.0
        feather = feather[..., None]
        src_f = detail_source.astype(np.float32)
        for sigma, w in zip(scales, weights):
            blurred = cv2.GaussianBlur(detail_source, (0, 0), sigmaX=sigma).astype(np.float32)
            high = src_f - blurred
            out += detail_strength * w * high * feather
        return np.clip(out, 0, 255).astype(np.uint8)

    def _graft_source_pixels(self, base, source_img, source_face, target_face,
                             detail, preserve_source_tone):
        """Photoshop-style transplant via multi-scale Laplacian frequency
        blend + Poisson seamless clone. The FaceFusion HD / Roop-Unleashed
        HD pipeline.

        Pipeline
        --------
        1.  inswapper has already produced `base` (a pose / expression /
            lighting-adapted version of the target). GFPGAN has upscaled
            `base` to the 2× HD canvas.
        2.  Affine-warp the FULL-resolution source image onto target's
            facial landmarks via dense MediaPipe FaceMesh (478 pts) →
            6-DOF least-squares fit. This puts the source's actual
            skin / eye / lip / hair pixels at the correct positions in
            target space.
        3.  LAB-space mean/std color transfer of the warped source to the
            base inside the face mask — so the source's color stats match
            the target's lighting / white balance.
        4.  Multi-scale Laplacian frequency blend: ADD the high-frequency
            content of the warped source onto `base` inside the face mask
            at σ=1.2, σ=3.0, σ=6.0. base gives the structure (pose,
            position, lighting); warped source gives the texture (pores,
            eye glints, eyelashes, hair strands). No alpha mixing of two
            full faces, so no "double-image" ghosting possible.
        5.  Poisson seamless-clone the graft back onto base inside the
            face mask. This kills any patch line at the boundary.
        """
        h, w = base.shape[:2]

        # Dense MediaPipe FaceMesh landmarks (478 points) on both images.
        src_dense = self._dense_landmarks(source_img, source_face.bbox)
        tgt_dense = self._dense_landmarks(base, target_face.bbox)
        use_dense = (
            src_dense is not None and tgt_dense is not None
            and len(src_dense) == len(tgt_dense)
        )

        # Step 2: affine warp source → target. Use the dense 6-DOF fit when
        # MediaPipe found both, else fall back to the 5-keypoint similarity
        # transform from insightface.
        if use_dense:
            M, _ = cv2.estimateAffine2D(src_dense, tgt_dense, method=cv2.LMEDS)
        else:
            M, _ = cv2.estimateAffinePartial2D(
                source_face.kps.astype(np.float32),
                target_face.kps.astype(np.float32),
                method=cv2.LMEDS,
            )
        if M is None:
            return base

        warped_source = cv2.warpAffine(
            source_img, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Mask: BiSeNet skin region of the target, intersected with the
        # warped source's skin region so we never graft from outside the
        # source face into the target. Then morphologically closed and
        # eroded so the boundary lands a few pixels inside the face.
        target_skin = self._skin_mask(base, target_face)
        if target_skin is None:
            target_skin = self._hull_fallback_mask(target_face, base.shape)
        if target_skin.max() == 0:
            return base
        src_skin_raw = self._skin_mask(source_img, source_face)
        if src_skin_raw is None:
            src_skin_raw = self._hull_fallback_mask(source_face, source_img.shape)
        warped_src_skin = cv2.warpAffine(
            src_skin_raw, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = cv2.bitwise_and(target_skin, warped_src_skin)
        if mask.max() < 32:
            mask = target_skin
        face_w_px = max(1, int(target_face.bbox[2] - target_face.bbox[0]))
        close_k = max(5, int(face_w_px * 0.02)) | 1
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8),
        )
        erode_edge = max(3, int(face_w_px * 0.012))
        mask = cv2.erode(mask, np.ones((erode_edge, erode_edge), np.uint8))

        # Step 3: LAB color-transfer of warped source to base under the mask
        # (skippable when "preserve source tone" is on).
        if not preserve_source_tone:
            cc_source = self._color_transfer_lab(warped_source, base, mask)
        else:
            cc_source = warped_source

        # Step 4: multi-scale Laplacian-pyramid graft. base + Σ w_i * HP_i(src)
        # where HP_i is high-pass at scale i. Strength is `detail`.
        graft = self._multi_band_blend(
            base, cc_source, mask,
            scales=(1.2, 3.0, 6.0),
            weights=(0.55, 0.35, 0.20),
            detail_strength=float(np.clip(detail, 0.0, 1.5)),
        )

        # Step 5: Poisson seamless-clone the graft back into base. Boundary
        # patch line disappears. Wrap in try because cv2.seamlessClone can
        # fail on tiny / degenerate masks.
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return graft
        cx = max(1, min(w - 2, int((xs.min() + xs.max()) / 2)))
        cy = max(1, min(h - 2, int((ys.min() + ys.max()) / 2)))
        try:
            result = cv2.seamlessClone(
                graft, base, mask, (cx, cy), cv2.NORMAL_CLONE,
            )
        except cv2.error:
            # Fallback: feathered alpha composite.
            feather_k = max(11, int(face_w_px * 0.02) | 1)
            f = cv2.GaussianBlur(mask, (feather_k, feather_k), 0)
            alpha = (f.astype(np.float32) / 255.0)[..., None]
            result = (graft.astype(np.float32) * alpha
                      + base.astype(np.float32) * (1 - alpha))
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def _hull_fallback_mask(self, face, shape):
        """Convex hull from insightface 106 landmarks. Used only when BiSeNet
        is unavailable or returns nothing."""
        h, w = shape[:2]
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            pts = face.landmark_2d_106.astype(np.int32)
        else:
            b = face.bbox.astype(np.int32)
            pts = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask
