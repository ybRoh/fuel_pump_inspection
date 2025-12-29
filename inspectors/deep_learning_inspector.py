"""
딥러닝 기반 결함 검사 모듈
Deep Learning Inspection Module for Fuel Pump

지원 모델:
- YOLOv8 (객체 검출 / 결함 검출)
- Classification (이종품 분류)
- Anomaly Detection (이상 탐지)

지원 추론 엔진:
- Hailo-8L (라즈베리파이 AI 키트)
- ONNX Runtime
- PyTorch (CPU/GPU)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import os
import time


class InferenceEngine(Enum):
    """추론 엔진 종류"""
    HAILO = "hailo"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    OPENCV_DNN = "opencv_dnn"


@dataclass
class Detection:
    """검출 결과 데이터 클래스"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    mask: Optional[np.ndarray] = None  # 세그멘테이션 마스크

    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox
        }


@dataclass
class ClassificationResult:
    """분류 결과 데이터 클래스"""
    class_id: int
    class_name: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'all_scores': self.all_scores
        }


@dataclass
class DLInspectionResult:
    """딥러닝 검사 결과"""
    is_ok: bool
    detections: List[Detection]
    classification: Optional[ClassificationResult]
    anomaly_score: float
    inference_time_ms: float
    message: str

    def to_dict(self) -> dict:
        return {
            'is_ok': self.is_ok,
            'detections': [d.to_dict() for d in self.detections],
            'classification': self.classification.to_dict() if self.classification else None,
            'anomaly_score': self.anomaly_score,
            'inference_time_ms': self.inference_time_ms,
            'message': self.message
        }


class DeepLearningInspector:
    """딥러닝 기반 검사 클래스"""

    # 기본 결함 클래스
    DEFAULT_DEFECT_CLASSES = [
        'scratch',      # 스크래치
        'dent',         # 찍힘
        'crack',        # 균열
        'contamination', # 이물질
        'burr',         # 버
        'missing_part', # 부품 누락
        'wrong_part',   # 이종품
    ]

    DEFAULT_DEFECT_CLASSES_KR = {
        'scratch': '스크래치',
        'dent': '찍힘',
        'crack': '균열',
        'contamination': '이물질',
        'burr': '버(Burr)',
        'missing_part': '부품누락',
        'wrong_part': '이종품',
        'ok': '정상',
        'ng': '불량'
    }

    def __init__(self, config: dict):
        """
        Args:
            config: 딥러닝 검사 설정
                - enabled: 사용 여부
                - engine: 추론 엔진 (hailo, onnx, pytorch)
                - detection_model: 결함 검출 모델 경로
                - classification_model: 분류 모델 경로
                - confidence_threshold: 신뢰도 임계값
                - defect_classes: 결함 클래스 목록
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.engine_type = config.get('engine', 'onnx')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.input_size = config.get('input_size', [640, 640])

        # 결함 클래스
        self.defect_classes = config.get('defect_classes', self.DEFAULT_DEFECT_CLASSES)

        # 모델 경로
        self.detection_model_path = config.get('detection_model', None)
        self.classification_model_path = config.get('classification_model', None)
        self.anomaly_model_path = config.get('anomaly_model', None)

        # 추론 엔진
        self.detection_engine = None
        self.classification_engine = None
        self.anomaly_engine = None

        # 초기화
        self._initialized = False
        if self.enabled:
            self._initialize_engines()

    def _initialize_engines(self) -> None:
        """추론 엔진 초기화"""
        print(f"\n[딥러닝 검사 모듈 초기화]")
        print(f"  추론 엔진: {self.engine_type}")

        # 결함 검출 모델
        if self.detection_model_path and os.path.exists(self.detection_model_path):
            self.detection_engine = self._load_model(
                self.detection_model_path, 'detection'
            )
            print(f"  결함 검출 모델: {self.detection_model_path}")

        # 분류 모델
        if self.classification_model_path and os.path.exists(self.classification_model_path):
            self.classification_engine = self._load_model(
                self.classification_model_path, 'classification'
            )
            print(f"  분류 모델: {self.classification_model_path}")

        # 이상 탐지 모델
        if self.anomaly_model_path and os.path.exists(self.anomaly_model_path):
            self.anomaly_engine = self._load_model(
                self.anomaly_model_path, 'anomaly'
            )
            print(f"  이상탐지 모델: {self.anomaly_model_path}")

        self._initialized = True
        print(f"  딥러닝 모듈 초기화: OK")

    def _load_model(self, model_path: str, model_type: str) -> Any:
        """
        모델 로드

        Args:
            model_path: 모델 파일 경로
            model_type: 모델 종류 (detection, classification, anomaly)

        Returns:
            로드된 모델 객체
        """
        ext = os.path.splitext(model_path)[1].lower()

        if self.engine_type == 'hailo' or ext == '.hef':
            return self._load_hailo_model(model_path)
        elif self.engine_type == 'onnx' or ext == '.onnx':
            return self._load_onnx_model(model_path)
        elif self.engine_type == 'pytorch' or ext in ['.pt', '.pth']:
            return self._load_pytorch_model(model_path)
        elif ext in ['.xml', '.bin']:
            return self._load_openvino_model(model_path)
        else:
            # OpenCV DNN (범용)
            return self._load_opencv_dnn_model(model_path)

    def _load_hailo_model(self, model_path: str) -> Any:
        """Hailo 모델 로드"""
        try:
            from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
            from hailo_platform import HEF

            hef = HEF(model_path)

            # VDevice 생성
            params = VDevice.create_params()
            self.hailo_vdevice = VDevice(params)

            # 네트워크 그룹 설정
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.hailo_network_group = self.hailo_vdevice.configure(hef, configure_params)[0]

            # 입출력 정보
            self.hailo_input_vstreams_info = self.hailo_network_group.get_input_vstream_infos()
            self.hailo_output_vstreams_info = self.hailo_network_group.get_output_vstream_infos()

            return self.hailo_network_group

        except ImportError:
            print("  Warning: Hailo SDK not available, falling back to ONNX")
            return self._load_onnx_model(model_path.replace('.hef', '.onnx'))
        except Exception as e:
            print(f"  Hailo 모델 로드 실패: {e}")
            return None

    def _load_onnx_model(self, model_path: str) -> Any:
        """ONNX 모델 로드"""
        try:
            import onnxruntime as ort

            # 세션 옵션
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # 프로바이더 (CPU 또는 GPU)
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            session = ort.InferenceSession(model_path, sess_options, providers=providers)
            return session

        except ImportError:
            print("  Warning: onnxruntime not installed")
            return None
        except Exception as e:
            print(f"  ONNX 모델 로드 실패: {e}")
            return None

    def _load_pytorch_model(self, model_path: str) -> Any:
        """PyTorch 모델 로드"""
        try:
            import torch

            model = torch.load(model_path, map_location='cpu')
            model.eval()
            return model

        except ImportError:
            print("  Warning: PyTorch not installed")
            return None
        except Exception as e:
            print(f"  PyTorch 모델 로드 실패: {e}")
            return None

    def _load_opencv_dnn_model(self, model_path: str) -> Any:
        """OpenCV DNN 모델 로드"""
        try:
            net = cv2.dnn.readNet(model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return net
        except Exception as e:
            print(f"  OpenCV DNN 모델 로드 실패: {e}")
            return None

    def _load_openvino_model(self, model_path: str) -> Any:
        """OpenVINO 모델 로드"""
        try:
            from openvino.runtime import Core

            ie = Core()
            model = ie.read_model(model_path)
            compiled_model = ie.compile_model(model, "CPU")
            return compiled_model

        except ImportError:
            print("  Warning: OpenVINO not installed")
            return None
        except Exception as e:
            print(f"  OpenVINO 모델 로드 실패: {e}")
            return None

    def inspect(self, image: np.ndarray) -> DLInspectionResult:
        """
        딥러닝 기반 검사 수행

        Args:
            image: BGR 형식 이미지

        Returns:
            딥러닝 검사 결과
        """
        start_time = time.time()

        detections = []
        classification = None
        anomaly_score = 0.0

        # 1. 결함 검출
        if self.detection_engine is not None:
            detections = self._run_detection(image)

        # 2. 분류
        if self.classification_engine is not None:
            classification = self._run_classification(image)

        # 3. 이상 탐지
        if self.anomaly_engine is not None:
            anomaly_score = self._run_anomaly_detection(image)

        # 검사 시간
        inference_time = (time.time() - start_time) * 1000

        # 판정
        is_ok = len(detections) == 0 and anomaly_score < 0.5

        # 메시지 생성
        if is_ok:
            message = "정상"
        else:
            defect_names = list(set(d.class_name for d in detections))
            if defect_names:
                defect_names_kr = [self.DEFAULT_DEFECT_CLASSES_KR.get(n, n) for n in defect_names]
                message = f"결함 검출: {', '.join(defect_names_kr)}"
            elif anomaly_score >= 0.5:
                message = f"이상 탐지 (점수: {anomaly_score:.2f})"
            else:
                message = "불량"

        return DLInspectionResult(
            is_ok=is_ok,
            detections=detections,
            classification=classification,
            anomaly_score=anomaly_score,
            inference_time_ms=inference_time,
            message=message
        )

    def _run_detection(self, image: np.ndarray) -> List[Detection]:
        """결함 검출 실행"""
        detections = []

        # 전처리
        input_tensor = self._preprocess_detection(image)

        # 추론
        if self.engine_type == 'hailo':
            outputs = self._infer_hailo(self.detection_engine, input_tensor)
        elif self.engine_type == 'onnx':
            outputs = self._infer_onnx(self.detection_engine, input_tensor)
        else:
            outputs = self._infer_opencv_dnn(self.detection_engine, input_tensor)

        # 후처리 (YOLOv8 형식)
        if outputs is not None:
            detections = self._postprocess_yolo(outputs, image.shape)

        return detections

    def _preprocess_detection(self, image: np.ndarray) -> np.ndarray:
        """검출용 전처리"""
        h, w = self.input_size

        # 리사이즈
        resized = cv2.resize(image, (w, h))

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 정규화 [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # NCHW 형식으로 변환
        transposed = np.transpose(normalized, (2, 0, 1))

        # 배치 차원 추가
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def _infer_onnx(self, session, input_tensor: np.ndarray) -> np.ndarray:
        """ONNX 추론"""
        if session is None:
            return None

        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_tensor})
            return outputs[0]
        except Exception as e:
            print(f"ONNX 추론 오류: {e}")
            return None

    def _infer_hailo(self, network_group, input_tensor: np.ndarray) -> np.ndarray:
        """Hailo 추론"""
        if network_group is None:
            return None

        try:
            from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

            # 입출력 스트림 파라미터
            input_vstream_params = InputVStreamParams.make_from_network_group(
                network_group, quantized=False
            )
            output_vstream_params = OutputVStreamParams.make_from_network_group(
                network_group, quantized=False
            )

            # 추론 실행
            with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
                input_data = {self.hailo_input_vstreams_info[0].name: input_tensor}
                results = pipeline.infer(input_data)

            # 출력 추출
            output_name = self.hailo_output_vstreams_info[0].name
            return results[output_name]

        except Exception as e:
            print(f"Hailo 추론 오류: {e}")
            return None

    def _infer_opencv_dnn(self, net, input_tensor: np.ndarray) -> np.ndarray:
        """OpenCV DNN 추론"""
        if net is None:
            return None

        try:
            blob = cv2.dnn.blobFromImage(
                input_tensor[0].transpose(1, 2, 0),
                1/255.0, tuple(self.input_size), (0, 0, 0),
                swapRB=True, crop=False
            )
            net.setInput(blob)
            outputs = net.forward()
            return outputs
        except Exception as e:
            print(f"OpenCV DNN 추론 오류: {e}")
            return None

    def _postprocess_yolo(self, outputs: np.ndarray,
                          original_shape: Tuple[int, int, int]) -> List[Detection]:
        """YOLOv8 출력 후처리"""
        detections = []

        if outputs is None:
            return detections

        orig_h, orig_w = original_shape[:2]
        input_h, input_w = self.input_size

        # YOLOv8 출력 형식: [batch, num_classes+4, num_detections]
        # 또는 [batch, num_detections, num_classes+4]
        if len(outputs.shape) == 3:
            if outputs.shape[1] < outputs.shape[2]:
                outputs = outputs.transpose(0, 2, 1)
            outputs = outputs[0]  # 배치 제거

        boxes = []
        scores = []
        class_ids = []

        for detection in outputs:
            # x_center, y_center, width, height, class_scores...
            if len(detection) > 4:
                x_center, y_center, width, height = detection[:4]
                class_scores = detection[4:]

                max_score = np.max(class_scores)
                class_id = np.argmax(class_scores)

                if max_score >= self.confidence_threshold:
                    # 좌표 변환
                    x = int((x_center - width / 2) * orig_w / input_w)
                    y = int((y_center - height / 2) * orig_h / input_h)
                    w = int(width * orig_w / input_w)
                    h = int(height * orig_h / input_h)

                    boxes.append([x, y, w, h])
                    scores.append(float(max_score))
                    class_ids.append(int(class_id))

        # NMS 적용
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores,
                                        self.confidence_threshold,
                                        self.nms_threshold)

            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                class_name = self.defect_classes[class_ids[idx]] \
                    if class_ids[idx] < len(self.defect_classes) else f"class_{class_ids[idx]}"

                detections.append(Detection(
                    class_id=class_ids[idx],
                    class_name=class_name,
                    confidence=scores[idx],
                    bbox=tuple(boxes[idx])
                ))

        return detections

    def _run_classification(self, image: np.ndarray) -> ClassificationResult:
        """분류 실행"""
        # 전처리
        input_tensor = self._preprocess_classification(image)

        # 추론
        if self.engine_type == 'onnx':
            outputs = self._infer_onnx(self.classification_engine, input_tensor)
        else:
            outputs = None

        if outputs is None:
            return None

        # Softmax
        scores = self._softmax(outputs[0])
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        # 클래스 이름
        class_names = self.config.get('part_classes', ['Type-A', 'Type-B', 'Type-C'])
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

        # 전체 점수
        all_scores = {class_names[i]: float(scores[i])
                      for i in range(min(len(scores), len(class_names)))}

        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            all_scores=all_scores
        )

    def _preprocess_classification(self, image: np.ndarray) -> np.ndarray:
        """분류용 전처리"""
        size = self.config.get('classification_input_size', [224, 224])

        resized = cv2.resize(image, tuple(size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # ImageNet 정규화
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0).astype(np.float32)

        return batched

    def _run_anomaly_detection(self, image: np.ndarray) -> float:
        """이상 탐지 실행"""
        # 전처리
        input_tensor = self._preprocess_classification(image)

        # 추론
        if self.engine_type == 'onnx':
            outputs = self._infer_onnx(self.anomaly_engine, input_tensor)
        else:
            outputs = None

        if outputs is None:
            return 0.0

        # 이상 점수 (0~1, 높을수록 이상)
        anomaly_score = float(np.clip(outputs[0][0], 0, 1))

        return anomaly_score

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 함수"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def visualize(self, image: np.ndarray,
                  result: DLInspectionResult) -> np.ndarray:
        """
        검출 결과 시각화

        Args:
            image: 원본 이미지
            result: 검사 결과

        Returns:
            시각화된 이미지
        """
        display = image.copy()

        # 색상 맵
        colors = {
            'scratch': (0, 0, 255),       # 빨강
            'dent': (0, 165, 255),         # 주황
            'crack': (0, 0, 139),          # 진한빨강
            'contamination': (0, 255, 255), # 노랑
            'burr': (255, 0, 255),          # 마젠타
            'missing_part': (255, 0, 0),    # 파랑
            'wrong_part': (128, 0, 128),    # 보라
        }

        # 검출 결과 표시
        for det in result.detections:
            x, y, w, h = det.bbox
            color = colors.get(det.class_name, (0, 255, 0))

            # 바운딩 박스
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

            # 라벨
            label_kr = self.DEFAULT_DEFECT_CLASSES_KR.get(det.class_name, det.class_name)
            label = f"{label_kr} {det.confidence:.0%}"

            # 라벨 배경
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x, y-text_h-10), (x+text_w+10, y), color, -1)
            cv2.putText(display, label, (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 상태 표시
        color = (0, 255, 0) if result.is_ok else (0, 0, 255)
        status = "OK" if result.is_ok else "NG"
        cv2.putText(display, f"[DL] {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 추론 시간
        cv2.putText(display, f"Inference: {result.inference_time_ms:.1f}ms",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 분류 결과
        if result.classification:
            cv2.putText(display,
                       f"Part: {result.classification.class_name} ({result.classification.confidence:.0%})",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        return display

    @property
    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self._initialized and (
            self.detection_engine is not None or
            self.classification_engine is not None or
            self.anomaly_engine is not None
        )
