"""
Hailo AI 가속기 유틸리티
Hailo-8L Utility Module for Raspberry Pi AI Kit

기능:
- Hailo 장치 상태 확인
- 모델 변환 (ONNX → HEF)
- 실시간 추론 파이프라인
"""

import os
import subprocess
from typing import Optional, Dict, List, Any
import numpy as np

# Hailo SDK 확인
try:
    from hailo_platform import VDevice, HEF
    from hailo_platform import HailoStreamInterface, ConfigureParams
    from hailo_platform import InputVStreamParams, OutputVStreamParams, InferVStreams
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


def check_hailo_available() -> bool:
    """Hailo 장치 사용 가능 여부 확인"""
    if not HAILO_AVAILABLE:
        return False

    try:
        vdevice = VDevice()
        vdevice.release()
        return True
    except Exception:
        return False


def get_hailo_info() -> Dict[str, Any]:
    """Hailo 장치 정보 조회"""
    info = {
        'available': False,
        'sdk_installed': HAILO_AVAILABLE,
        'device_id': None,
        'architecture': None,
    }

    if not HAILO_AVAILABLE:
        return info

    try:
        vdevice = VDevice()
        info['available'] = True
        info['device_id'] = vdevice.get_physical_devices()[0].device_id()
        info['architecture'] = 'hailo8l'
        vdevice.release()
    except Exception as e:
        info['error'] = str(e)

    return info


class HailoInferenceEngine:
    """Hailo 추론 엔진 클래스"""

    def __init__(self, hef_path: str):
        """
        Args:
            hef_path: HEF 모델 파일 경로
        """
        if not HAILO_AVAILABLE:
            raise RuntimeError("Hailo SDK not installed")

        self.hef_path = hef_path
        self.hef = None
        self.vdevice = None
        self.network_group = None
        self.input_vstream_info = None
        self.output_vstream_info = None

        self._load_model()

    def _load_model(self) -> None:
        """모델 로드"""
        self.hef = HEF(self.hef_path)

        # 가상 장치 생성
        params = VDevice.create_params()
        self.vdevice = VDevice(params)

        # 네트워크 설정
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]

        # 입출력 정보
        self.input_vstream_info = self.network_group.get_input_vstream_infos()
        self.output_vstream_info = self.network_group.get_output_vstream_infos()

        print(f"Hailo 모델 로드: {self.hef_path}")
        print(f"  입력: {[info.name for info in self.input_vstream_info]}")
        print(f"  출력: {[info.name for info in self.output_vstream_info]}")

    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        추론 실행

        Args:
            input_data: 입력 텐서 (NCHW 형식)

        Returns:
            출력 텐서 딕셔너리
        """
        # 입출력 스트림 파라미터
        input_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False
        )
        output_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False
        )

        # 추론
        with InferVStreams(self.network_group, input_params, output_params) as pipeline:
            input_dict = {self.input_vstream_info[0].name: input_data}
            results = pipeline.infer(input_dict)

        return results

    def get_input_shape(self) -> tuple:
        """입력 형상 반환"""
        return self.input_vstream_info[0].shape

    def get_output_shape(self) -> tuple:
        """출력 형상 반환"""
        return self.output_vstream_info[0].shape

    def release(self) -> None:
        """리소스 해제"""
        if self.vdevice:
            self.vdevice.release()


def convert_onnx_to_hef(onnx_path: str, output_dir: str,
                        hw_arch: str = 'hailo8l') -> Optional[str]:
    """
    ONNX 모델을 Hailo HEF로 변환

    Args:
        onnx_path: ONNX 모델 경로
        output_dir: 출력 디렉토리
        hw_arch: 하드웨어 아키텍처 (hailo8l, hailo8)

    Returns:
        변환된 HEF 파일 경로 또는 None
    """
    try:
        # Hailo Model Zoo CLI 사용
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        hef_path = os.path.join(output_dir, f"{model_name}.hef")

        cmd = [
            'hailo', 'compile', onnx_path,
            '--hw-arch', hw_arch,
            '--output-dir', output_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(hef_path):
            print(f"HEF 변환 성공: {hef_path}")
            return hef_path
        else:
            print(f"HEF 변환 실패: {result.stderr}")
            return None

    except Exception as e:
        print(f"HEF 변환 오류: {e}")
        return None


def download_pretrained_model(model_name: str, output_dir: str) -> Optional[str]:
    """
    사전학습 모델 다운로드 (Hailo Model Zoo)

    Args:
        model_name: 모델 이름 (yolov8s, mobilenet_v2 등)
        output_dir: 출력 디렉토리

    Returns:
        다운로드된 모델 경로
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Hailo Model Zoo에서 다운로드
        cmd = [
            'hailo', 'model-zoo', 'download',
            model_name,
            '--hw-arch', 'hailo8l',
            '--output-dir', output_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            hef_path = os.path.join(output_dir, f"{model_name}.hef")
            if os.path.exists(hef_path):
                print(f"모델 다운로드 성공: {hef_path}")
                return hef_path

        print(f"모델 다운로드 실패: {result.stderr}")
        return None

    except Exception as e:
        print(f"모델 다운로드 오류: {e}")
        return None


# 사용 예시
if __name__ == "__main__":
    print("=== Hailo 장치 정보 ===")
    info = get_hailo_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
