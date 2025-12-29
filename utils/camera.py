"""
카메라 관리 모듈
Camera Management Module for Fuel Pump Inspection

지원 카메라:
- Raspberry Pi Camera (picamera2)
- USB Camera (OpenCV)
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod

# Pi Camera 지원 확인
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


class BaseCamera(ABC):
    """카메라 추상 베이스 클래스"""

    @abstractmethod
    def start(self) -> None:
        """카메라 시작"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """카메라 정지"""
        pass

    @abstractmethod
    def capture(self) -> np.ndarray:
        """이미지 캡처"""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """카메라 열림 상태 확인"""
        pass


class PiCamera(BaseCamera):
    """Raspberry Pi Camera 클래스"""

    def __init__(self, camera_id: int = 0,
                 resolution: Tuple[int, int] = (1920, 1080),
                 exposure: int = 10000):
        """
        Args:
            camera_id: 카메라 ID
            resolution: 해상도 (width, height)
            exposure: 노출 시간 (마이크로초)
        """
        if not PICAMERA_AVAILABLE:
            raise RuntimeError("picamera2 not available")

        self.camera_id = camera_id
        self.resolution = resolution
        self.exposure = exposure
        self.camera: Optional[Picamera2] = None
        self._is_started = False

    def start(self) -> None:
        """카메라 시작"""
        if self._is_started:
            return

        self.camera = Picamera2(self.camera_id)

        config = self.camera.create_still_configuration(
            main={"size": self.resolution}
        )
        self.camera.configure(config)

        # 노출 설정
        self.camera.set_controls({"ExposureTime": self.exposure})

        self.camera.start()
        self._is_started = True
        print(f"Pi Camera {self.camera_id} started")

    def stop(self) -> None:
        """카메라 정지"""
        if self.camera and self._is_started:
            self.camera.stop()
            self._is_started = False
            print(f"Pi Camera {self.camera_id} stopped")

    def capture(self) -> np.ndarray:
        """
        이미지 캡처

        Returns:
            BGR 형식의 이미지
        """
        if not self._is_started:
            self.start()

        frame = self.camera.capture_array()
        # RGB to BGR 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def is_opened(self) -> bool:
        """카메라 열림 상태 확인"""
        return self._is_started

    def set_exposure(self, exposure: int) -> None:
        """노출 설정"""
        self.exposure = exposure
        if self.camera and self._is_started:
            self.camera.set_controls({"ExposureTime": exposure})


class USBCamera(BaseCamera):
    """USB Camera 클래스 (OpenCV)"""

    def __init__(self, camera_id: int = 0,
                 resolution: Tuple[int, int] = (1920, 1080),
                 fps: int = 30):
        """
        Args:
            camera_id: 카메라 ID
            resolution: 해상도 (width, height)
            fps: 프레임 레이트
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.camera: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """카메라 시작"""
        if self.camera and self.camera.isOpened():
            return

        self.camera = cv2.VideoCapture(self.camera_id)

        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # 설정 적용
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)

        print(f"USB Camera {self.camera_id} started")

    def stop(self) -> None:
        """카메라 정지"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print(f"USB Camera {self.camera_id} stopped")

    def capture(self) -> np.ndarray:
        """
        이미지 캡처

        Returns:
            BGR 형식의 이미지
        """
        if not self.camera or not self.camera.isOpened():
            self.start()

        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        return frame

    def is_opened(self) -> bool:
        """카메라 열림 상태 확인"""
        return self.camera is not None and self.camera.isOpened()


class CameraManager:
    """카메라 관리 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 카메라 설정 딕셔너리
        """
        self.config = config
        self.cameras = {}

    def add_camera(self, name: str, camera_config: dict,
                   use_picamera: bool = True) -> None:
        """
        카메라 추가

        Args:
            name: 카메라 이름
            camera_config: 카메라 설정
            use_picamera: Pi Camera 사용 여부
        """
        camera_id = camera_config.get('id', 0)
        resolution = tuple(camera_config.get('resolution', [1920, 1080]))
        exposure = camera_config.get('exposure', 10000)
        fps = camera_config.get('fps', 30)

        if use_picamera and PICAMERA_AVAILABLE:
            camera = PiCamera(camera_id, resolution, exposure)
        else:
            camera = USBCamera(camera_id, resolution, fps)

        self.cameras[name] = camera

    def start_all(self) -> None:
        """모든 카메라 시작"""
        for name, camera in self.cameras.items():
            try:
                camera.start()
            except Exception as e:
                print(f"Failed to start camera '{name}': {e}")

    def stop_all(self) -> None:
        """모든 카메라 정지"""
        for name, camera in self.cameras.items():
            try:
                camera.stop()
            except Exception as e:
                print(f"Failed to stop camera '{name}': {e}")

    def capture(self, name: str) -> np.ndarray:
        """
        지정된 카메라에서 이미지 캡처

        Args:
            name: 카메라 이름

        Returns:
            BGR 형식의 이미지
        """
        if name not in self.cameras:
            raise ValueError(f"Camera '{name}' not found")

        return self.cameras[name].capture()

    def get_camera(self, name: str) -> Optional[BaseCamera]:
        """카메라 객체 반환"""
        return self.cameras.get(name)

    def list_cameras(self) -> list:
        """등록된 카메라 목록 반환"""
        return list(self.cameras.keys())


def detect_available_cameras(max_cameras: int = 5) -> list:
    """
    사용 가능한 카메라 검색

    Args:
        max_cameras: 검색할 최대 카메라 수

    Returns:
        사용 가능한 카메라 ID 리스트
    """
    available = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()

    return available
