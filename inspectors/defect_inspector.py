"""
외관 결함 검사 모듈
Defect Inspection Module for Fuel Pump

검출 가능한 결함:
- 스크래치 (Scratch)
- 찍힘/덴트 (Dent)
- 균열 (Crack)
- 버/사출불량 (Burr/Injection Defect)
- 이물질 (Contamination)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class DefectType(Enum):
    """결함 유형 정의"""
    SCRATCH = "스크래치"
    DENT = "찍힘"
    CRACK = "균열"
    BURR = "버(Burr)"
    CONTAMINATION = "이물질"
    INJECTION_DEFECT = "사출불량"


@dataclass
class Defect:
    """결함 정보 데이터 클래스"""
    type: DefectType
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    severity: str  # "경미", "보통", "심각"
    confidence: float

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'type': self.type.value,
            'bbox': self.bbox,
            'area': self.area,
            'severity': self.severity,
            'confidence': self.confidence
        }


class DefectInspector:
    """외관 결함 검사 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 검사 설정 딕셔너리
                - min_defect_area: 최소 결함 크기 (픽셀)
                - scratch_threshold: 스크래치 감도
                - crack_threshold: 균열 감도
                - dent_threshold: 찍힘 감도
        """
        self.config = config
        self.min_defect_area = config.get('min_defect_area', 50)
        self.scratch_threshold = config.get('scratch_threshold', 30)
        self.crack_threshold = config.get('crack_threshold', 25)
        self.dent_threshold = config.get('dent_threshold', 30)

        # 기준 이미지
        self.reference_image: Optional[np.ndarray] = None
        self.reference_edges: Optional[np.ndarray] = None

        # 특징점 검출기
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.ref_kp = None
        self.ref_desc = None

    def set_reference(self, image: np.ndarray) -> None:
        """
        정상품 기준 이미지 설정

        Args:
            image: BGR 형식의 기준 이미지
        """
        self.reference_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.reference_edges = cv2.Canny(self.reference_image, 50, 150)

        # 특징점 추출 (형상 비교용)
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(
            self.reference_image, None
        )

    def inspect(self, image: np.ndarray) -> Tuple[bool, List[Defect]]:
        """
        외관 결함 검사 수행

        Args:
            image: BGR 형식의 검사 이미지

        Returns:
            (합격 여부, 검출된 결함 리스트)
        """
        defects = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 스크래치 검출
        scratches = self._detect_scratches(gray)
        defects.extend(scratches)

        # 2. 찍힘/덴트 검출
        dents = self._detect_dents(gray)
        defects.extend(dents)

        # 3. 균열 검출
        cracks = self._detect_cracks(gray)
        defects.extend(cracks)

        # 4. 이물질 검출
        contaminations = self._detect_contamination(image)
        defects.extend(contaminations)

        # 5. 사출 불량 검출 (버, 미성형)
        injection_defects = self._detect_injection_defects(gray)
        defects.extend(injection_defects)

        # 판정
        is_ok = len(defects) == 0

        return is_ok, defects

    def _detect_scratches(self, gray: np.ndarray) -> List[Defect]:
        """
        스크래치 검출 - 선형 결함

        Args:
            gray: 그레이스케일 이미지

        Returns:
            검출된 스크래치 결함 리스트
        """
        defects = []

        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 선형 구조 검출 (가로/세로 커널)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        lines_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        lines_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
        lines = cv2.bitwise_or(lines_h, lines_v)

        # 컨투어 찾기
        contours, _ = cv2.findContours(
            lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_defect_area:
                x, y, w, h = cv2.boundingRect(cnt)

                # 선형성 확인 (가로세로 비율)
                aspect_ratio = max(w, h) / (min(w, h) + 1)
                if aspect_ratio > 3:  # 선형 형태
                    severity = self._calculate_severity(area)
                    defects.append(Defect(
                        type=DefectType.SCRATCH,
                        bbox=(x, y, w, h),
                        area=area,
                        severity=severity,
                        confidence=0.85
                    ))

        return defects

    def _detect_dents(self, gray: np.ndarray) -> List[Defect]:
        """
        찍힘/덴트 검출 - 국부적 밝기 변화

        Args:
            gray: 그레이스케일 이미지

        Returns:
            검출된 찍힘 결함 리스트
        """
        defects = []

        if self.reference_image is None:
            return defects

        # 크기 맞추기
        if gray.shape != self.reference_image.shape:
            gray = cv2.resize(gray, (self.reference_image.shape[1],
                                     self.reference_image.shape[0]))

        # 기준 이미지와 차이 비교
        diff = cv2.absdiff(gray, self.reference_image)

        # 형태학적 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        _, binary = cv2.threshold(diff, self.dent_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_defect_area:
                # 원형도 계산
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1)

                if circularity > 0.5:  # 원형에 가까운 결함
                    x, y, w, h = cv2.boundingRect(cnt)
                    severity = self._calculate_severity(area)
                    defects.append(Defect(
                        type=DefectType.DENT,
                        bbox=(x, y, w, h),
                        area=area,
                        severity=severity,
                        confidence=0.80
                    ))

        return defects

    def _detect_cracks(self, gray: np.ndarray) -> List[Defect]:
        """
        균열 검출 - 엣지 기반

        Args:
            gray: 그레이스케일 이미지

        Returns:
            검출된 균열 결함 리스트
        """
        defects = []

        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)

        if self.reference_edges is not None:
            # 크기 맞추기
            if edges.shape != self.reference_edges.shape:
                edges = cv2.resize(edges, (self.reference_edges.shape[1],
                                           self.reference_edges.shape[0]))
            # 기준 엣지와 비교하여 새로운 엣지 찾기
            new_edges = cv2.subtract(edges, self.reference_edges)
        else:
            new_edges = edges

        # 균열 특성: 불규칙한 선형 패턴
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        new_edges = cv2.morphologyEx(new_edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            arc_length = cv2.arcLength(cnt, False)

            if arc_length > 50:  # 일정 길이 이상의 균열
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append(Defect(
                    type=DefectType.CRACK,
                    bbox=(x, y, w, h),
                    area=area,
                    severity="심각",  # 균열은 항상 심각
                    confidence=0.75
                ))

        return defects

    def _detect_contamination(self, image: np.ndarray) -> List[Defect]:
        """
        이물질 검출 - 색상 이상

        Args:
            image: BGR 형식 이미지

        Returns:
            검출된 이물질 결함 리스트
        """
        defects = []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 연료펌프 정상 색상 범위 (검정/회색 계열)
        # 이 범위를 벗어나는 색상 = 이물질
        lower_normal = np.array([0, 0, 0])
        upper_normal = np.array([180, 50, 200])

        normal_mask = cv2.inRange(hsv, lower_normal, upper_normal)
        abnormal_mask = cv2.bitwise_not(normal_mask)

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 30:  # 작은 이물질도 검출
                x, y, w, h = cv2.boundingRect(cnt)
                severity = self._calculate_severity(area)
                defects.append(Defect(
                    type=DefectType.CONTAMINATION,
                    bbox=(x, y, w, h),
                    area=area,
                    severity=severity,
                    confidence=0.90
                ))

        return defects

    def _detect_injection_defects(self, gray: np.ndarray) -> List[Defect]:
        """
        사출 불량 검출 (버, 미성형)

        Args:
            gray: 그레이스케일 이미지

        Returns:
            검출된 사출 불량 결함 리스트
        """
        defects = []

        if self.reference_image is None or self.ref_desc is None:
            return defects

        # 특징점 매칭으로 형상 비교
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None:
            return defects

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        try:
            matches = bf.match(self.ref_desc, desc)
        except cv2.error:
            return defects

        # 매칭률 계산
        match_ratio = len(matches) / max(len(self.ref_kp), 1)

        if match_ratio < 0.7:  # 형상 불일치
            # 크기 맞추기
            if gray.shape != self.reference_image.shape:
                gray_resized = cv2.resize(gray, (self.reference_image.shape[1],
                                                  self.reference_image.shape[0]))
            else:
                gray_resized = gray

            # 윤곽선 비교로 불량 위치 특정
            edges = cv2.Canny(gray_resized, 50, 150)
            ref_edges = cv2.Canny(self.reference_image, 50, 150)

            diff = cv2.absdiff(edges, ref_edges)

            contours, _ = cv2.findContours(
                diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    defects.append(Defect(
                        type=DefectType.INJECTION_DEFECT,
                        bbox=(x, y, w, h),
                        area=area,
                        severity="보통",
                        confidence=0.70
                    ))

        return defects

    def _calculate_severity(self, area: float) -> str:
        """
        결함 크기에 따른 심각도 계산

        Args:
            area: 결함 면적 (픽셀)

        Returns:
            심각도 문자열 ("경미", "보통", "심각")
        """
        if area < 100:
            return "경미"
        elif area < 500:
            return "보통"
        else:
            return "심각"

    def visualize_defects(self, image: np.ndarray,
                          defects: List[Defect]) -> np.ndarray:
        """
        결함 시각화

        Args:
            image: 원본 이미지
            defects: 검출된 결함 리스트

        Returns:
            결함이 표시된 이미지
        """
        display = image.copy()

        colors = {
            DefectType.SCRATCH: (0, 0, 255),       # 빨강
            DefectType.DENT: (0, 165, 255),        # 주황
            DefectType.CRACK: (0, 0, 139),         # 진한 빨강
            DefectType.BURR: (255, 0, 255),        # 마젠타
            DefectType.CONTAMINATION: (0, 255, 255), # 노랑
            DefectType.INJECTION_DEFECT: (255, 0, 0), # 파랑
        }

        for defect in defects:
            x, y, w, h = defect.bbox
            color = colors.get(defect.type, (0, 0, 255))

            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

            label = f"{defect.type.value} ({defect.severity})"
            cv2.putText(display, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display
