"""
바코드 검사 모듈
Barcode Inspection Module for Fuel Pump

검사 항목:
- 바코드 부착 유무
- 바코드 위치 정확성
- 바코드 인식 가능 여부
- 바코드 데이터 유효성
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("Warning: pyzbar not installed. Install with: pip install pyzbar")


@dataclass
class BarcodeResult:
    """바코드 검사 결과 데이터 클래스"""
    detected: bool
    data: Optional[str]
    barcode_type: Optional[str]
    position: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    position_ok: bool
    readable: bool
    message: str

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'detected': self.detected,
            'data': self.data,
            'barcode_type': self.barcode_type,
            'position': self.position,
            'position_ok': self.position_ok,
            'readable': self.readable,
            'message': self.message
        }


class BarcodeInspector:
    """바코드 검사 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 검사 설정 딕셔너리
                - roi: 바코드 예상 위치 [x, y, w, h]
                - required_prefix: 바코드 시작 문자
                - min_length: 최소 바코드 길이
                - position_tolerance: 위치 허용 오차
        """
        self.config = config
        self.expected_roi = config.get('roi', None)
        self.required_prefix = config.get('required_prefix', '')
        self.min_length = config.get('min_length', 8)
        self.position_tolerance = config.get('position_tolerance', 50)

        if not PYZBAR_AVAILABLE:
            print("Warning: Barcode inspection disabled - pyzbar not available")

    def inspect(self, image: np.ndarray) -> Tuple[bool, BarcodeResult]:
        """
        바코드 검사 수행

        Args:
            image: BGR 형식의 검사 이미지

        Returns:
            (합격 여부, 바코드 검사 결과)
        """
        if not PYZBAR_AVAILABLE:
            return True, BarcodeResult(
                detected=False,
                data=None,
                barcode_type=None,
                position=None,
                position_ok=True,
                readable=False,
                message="pyzbar 미설치 - 검사 생략"
            )

        # 1. 바코드 검출 시도
        barcodes = pyzbar.decode(image)

        if not barcodes:
            # 전처리 후 재시도
            processed = self._preprocess(image)
            barcodes = pyzbar.decode(processed)

        if not barcodes:
            return False, BarcodeResult(
                detected=False,
                data=None,
                barcode_type=None,
                position=None,
                position_ok=False,
                readable=False,
                message="바코드 미검출 - 부착 확인 필요"
            )

        barcode = barcodes[0]
        data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # 바코드 위치 계산
        rect = barcode.rect
        position = (rect.left, rect.top, rect.width, rect.height)

        # 2. 위치 검사
        position_ok = self._check_position(barcode)

        # 3. 데이터 유효성 검사
        data_valid = self._validate_data(data)

        # 4. 판정
        is_ok = position_ok and data_valid

        message = "정상"
        if not position_ok:
            message = "바코드 위치 불량"
        elif not data_valid:
            message = f"바코드 데이터 불량: {data}"

        return is_ok, BarcodeResult(
            detected=True,
            data=data,
            barcode_type=barcode_type,
            position=position,
            position_ok=position_ok,
            readable=True,
            message=message
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        바코드 인식률 향상을 위한 전처리

        Args:
            image: BGR 형식 이미지

        Returns:
            전처리된 그레이스케일 이미지
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE 적용 (대비 향상)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 샤프닝
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def _check_position(self, barcode) -> bool:
        """
        바코드 부착 위치 확인

        Args:
            barcode: pyzbar 바코드 객체

        Returns:
            위치 정상 여부
        """
        if self.expected_roi is None:
            return True

        # 바코드 중심점 계산
        if hasattr(barcode, 'polygon') and barcode.polygon:
            points = barcode.polygon
            cx = sum(p.x for p in points) / len(points)
            cy = sum(p.y for p in points) / len(points)
        else:
            rect = barcode.rect
            cx = rect.left + rect.width / 2
            cy = rect.top + rect.height / 2

        # ROI 내부인지 확인
        rx, ry, rw, rh = self.expected_roi
        margin = self.position_tolerance

        in_roi = (rx - margin <= cx <= rx + rw + margin and
                  ry - margin <= cy <= ry + rh + margin)

        return in_roi

    def _validate_data(self, data: str) -> bool:
        """
        바코드 데이터 유효성 검사

        Args:
            data: 바코드 데이터 문자열

        Returns:
            유효성 여부
        """
        # 길이 검사
        if len(data) < self.min_length:
            return False

        # 접두어 검사
        if self.required_prefix and not data.startswith(self.required_prefix):
            return False

        return True

    def detect_all_barcodes(self, image: np.ndarray) -> List[dict]:
        """
        이미지에서 모든 바코드 검출

        Args:
            image: BGR 형식 이미지

        Returns:
            검출된 바코드 정보 리스트
        """
        if not PYZBAR_AVAILABLE:
            return []

        barcodes = pyzbar.decode(image)
        results = []

        for barcode in barcodes:
            rect = barcode.rect
            results.append({
                'data': barcode.data.decode('utf-8'),
                'type': barcode.type,
                'position': (rect.left, rect.top, rect.width, rect.height)
            })

        return results

    def visualize_barcode(self, image: np.ndarray,
                          result: BarcodeResult) -> np.ndarray:
        """
        바코드 검출 결과 시각화

        Args:
            image: 원본 이미지
            result: 바코드 검사 결과

        Returns:
            시각화된 이미지
        """
        display = image.copy()

        # 예상 ROI 표시 (파란색)
        if self.expected_roi:
            rx, ry, rw, rh = self.expected_roi
            cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 1)
            cv2.putText(display, "Expected ROI", (rx, ry-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 검출된 바코드 표시
        if result.detected and result.position:
            x, y, w, h = result.position
            color = (0, 255, 0) if result.position_ok else (0, 0, 255)

            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

            label = f"{result.data} ({result.barcode_type})"
            cv2.putText(display, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return display
