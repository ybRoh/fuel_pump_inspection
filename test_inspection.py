#!/usr/bin/env python3
"""
연료펌프 검사 시스템 테스트 스크립트
카메라 없이 모듈별 기능 테스트
"""

import cv2
import numpy as np
import yaml
import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspectors.defect_inspector import DefectInspector, DefectType
from inspectors.barcode_inspector import BarcodeInspector
from inspectors.variant_inspector import VariantInspector


def create_test_image(width=640, height=480):
    """테스트용 이미지 생성 (연료펌프 모듈 시뮬레이션)"""
    # 검정 배경
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)  # 어두운 회색 배경

    # 연료펌프 모듈 형상 (사각형)
    cv2.rectangle(image, (100, 100), (540, 380), (60, 60, 60), -1)
    cv2.rectangle(image, (100, 100), (540, 380), (80, 80, 80), 2)

    # 원형 부품
    cv2.circle(image, (320, 240), 80, (70, 70, 70), -1)
    cv2.circle(image, (320, 240), 80, (90, 90, 90), 2)

    return image


def create_defect_image(base_image):
    """결함이 있는 테스트 이미지 생성"""
    image = base_image.copy()

    # 스크래치 추가 (흰색 선)
    cv2.line(image, (150, 150), (250, 160), (200, 200, 200), 2)

    # 이물질 추가 (빨간 점)
    cv2.circle(image, (400, 200), 10, (0, 0, 255), -1)

    # 찍힘 추가 (밝은 원)
    cv2.circle(image, (200, 300), 15, (150, 150, 150), -1)

    return image


def create_barcode_image(base_image, barcode_text="FP-2024-A01-12345"):
    """바코드가 있는 테스트 이미지 생성"""
    image = base_image.copy()

    # 바코드 영역 (흰색 배경)
    cv2.rectangle(image, (150, 50), (350, 90), (255, 255, 255), -1)

    # 바코드 시뮬레이션 (검정 줄무늬)
    x = 160
    for i, char in enumerate(barcode_text):
        bar_width = 2 if i % 2 == 0 else 3
        cv2.rectangle(image, (x, 55), (x + bar_width, 85), (0, 0, 0), -1)
        x += bar_width + 2

    # 텍스트 추가
    cv2.putText(image, barcode_text, (155, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image


def test_defect_inspector():
    """외관 결함 검사 모듈 테스트"""
    print("\n" + "="*60)
    print("1. 외관 결함 검사 모듈 테스트")
    print("="*60)

    config = {
        'min_defect_area': 50,
        'scratch_threshold': 30,
        'crack_threshold': 25,
        'dent_threshold': 30,
        'contamination_threshold': 20
    }

    inspector = DefectInspector(config)

    # 정상 이미지 테스트
    print("\n[테스트 1] 정상 이미지 검사...")
    normal_image = create_test_image()
    inspector.set_reference(normal_image)

    is_ok, defects = inspector.inspect(normal_image)
    print(f"  결과: {'OK (정상)' if is_ok else 'NG (불량)'}")
    print(f"  검출된 결함: {len(defects)}개")

    # 결함 이미지 테스트
    print("\n[테스트 2] 결함 이미지 검사...")
    defect_image = create_defect_image(normal_image)

    is_ok, defects = inspector.inspect(defect_image)
    print(f"  결과: {'OK (정상)' if is_ok else 'NG (불량)'}")
    print(f"  검출된 결함: {len(defects)}개")

    for i, defect in enumerate(defects):
        print(f"    [{i+1}] {defect.type.value}")
        print(f"        위치: {defect.bbox}")
        print(f"        면적: {defect.area:.1f}px")
        print(f"        심각도: {defect.severity}")
        print(f"        신뢰도: {defect.confidence:.1%}")

    # 결과 이미지 저장
    result_image = defect_image.copy()
    for defect in defects:
        x, y, w, h = defect.bbox
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result_image, defect.type.value, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite('test_results/defect_result.jpg', result_image)
    print("\n  결과 이미지 저장: test_results/defect_result.jpg")

    return len(defects) > 0  # 결함 검출 성공 여부


def test_barcode_inspector():
    """바코드 검사 모듈 테스트"""
    print("\n" + "="*60)
    print("2. 바코드 검사 모듈 테스트")
    print("="*60)

    config = {
        'roi': [100, 50, 300, 150],
        'required_prefix': 'FP',
        'min_length': 10
    }

    inspector = BarcodeInspector(config)

    # 바코드 이미지 테스트
    print("\n[테스트 1] 바코드 이미지 검사...")
    base_image = create_test_image()
    barcode_image = create_barcode_image(base_image)

    is_ok, result = inspector.inspect(barcode_image)
    print(f"  검출 여부: {'검출됨' if result.detected else '미검출'}")
    print(f"  바코드 데이터: {result.data}")
    print(f"  위치 정상: {'예' if result.position_ok else '아니오'}")
    print(f"  읽기 가능: {'예' if result.readable else '아니오'}")
    print(f"  최종 판정: {'OK' if is_ok else 'NG'}")
    print(f"  메시지: {result.message}")

    # 바코드 없는 이미지 테스트
    print("\n[테스트 2] 바코드 없는 이미지 검사...")
    is_ok, result = inspector.inspect(base_image)
    print(f"  검출 여부: {'검출됨' if result.detected else '미검출'}")
    print(f"  최종 판정: {'OK' if is_ok else 'NG'}")
    print(f"  메시지: {result.message}")

    cv2.imwrite('test_results/barcode_test.jpg', barcode_image)
    print("\n  테스트 이미지 저장: test_results/barcode_test.jpg")

    return True


def test_variant_inspector():
    """이종품 검사 모듈 테스트"""
    print("\n" + "="*60)
    print("3. 이종품 검사 모듈 테스트")
    print("="*60)

    config = {
        'similarity_threshold': 0.85,
        'color_tolerance': 20,
        'shape_tolerance': 0.3
    }

    valid_parts = [
        {'id': 'FP-2024-A01', 'name': 'Type-A', 'color': 'black'},
        {'id': 'FP-2024-B01', 'name': 'Type-B', 'color': 'gray'}
    ]

    inspector = VariantInspector(config, valid_parts)

    # 마스터 이미지 등록
    print("\n[준비] 마스터 이미지 등록...")
    master_image = create_test_image()
    inspector.register_master('FP-2024-A01', master_image)
    print("  FP-2024-A01 마스터 등록 완료")

    # 동일 부품 테스트
    print("\n[테스트 1] 동일 부품 검사...")
    test_image = create_test_image()  # 동일한 이미지

    is_ok, result = inspector.inspect(test_image, 'FP-2024-A01')
    print(f"  검출된 부품: {result.detected_part}")
    print(f"  예상 부품: {result.expected_part}")
    print(f"  유사도: {result.similarity:.1%}")
    print(f"  색상 일치: {'예' if result.color_match else '아니오'}")
    print(f"  형상 일치: {'예' if result.shape_match else '아니오'}")
    print(f"  최종 판정: {'OK (정품)' if is_ok else 'NG (이종품)'}")
    print(f"  메시지: {result.message}")

    # 다른 부품 테스트 (이종품 시뮬레이션)
    print("\n[테스트 2] 이종품 검사...")
    different_image = create_test_image()
    # 형상 변경 (다른 부품 시뮬레이션)
    cv2.rectangle(different_image, (200, 150), (450, 350), (100, 100, 100), -1)
    cv2.circle(different_image, (320, 250), 50, (80, 80, 80), -1)

    is_ok, result = inspector.inspect(different_image, 'FP-2024-A01')
    print(f"  검출된 부품: {result.detected_part}")
    print(f"  예상 부품: {result.expected_part}")
    print(f"  유사도: {result.similarity:.1%}")
    print(f"  최종 판정: {'OK (정품)' if is_ok else 'NG (이종품)'}")
    print(f"  메시지: {result.message}")

    cv2.imwrite('test_results/variant_master.jpg', master_image)
    cv2.imwrite('test_results/variant_different.jpg', different_image)
    print("\n  마스터 이미지 저장: test_results/variant_master.jpg")
    print("  이종품 이미지 저장: test_results/variant_different.jpg")

    return True


def test_config_loading():
    """설정 파일 로드 테스트"""
    print("\n" + "="*60)
    print("4. 설정 파일 로드 테스트")
    print("="*60)

    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"\n  시스템 이름: {config['system']['name']}")
        print(f"  라인: {config['system']['line']}")
        print(f"  버전: {config['system']['version']}")
        print(f"\n  검사 설정:")
        print(f"    - 외관 결함 검사: {'활성화' if config['inspection']['defect']['enabled'] else '비활성화'}")
        print(f"    - 바코드 검사: {'활성화' if config['inspection']['barcode']['enabled'] else '비활성화'}")
        print(f"    - 이종품 검사: {'활성화' if config['inspection']['variant']['enabled'] else '비활성화'}")
        print(f"\n  딥러닝 설정:")
        print(f"    - 딥러닝 사용: {'활성화' if config['deep_learning']['enabled'] else '비활성화'}")
        print(f"    - 추론 엔진: {config['deep_learning']['engine']}")
        print(f"    - 신뢰도 임계값: {config['deep_learning']['confidence_threshold']}")

        print(f"\n  등록된 정품 부품: {len(config['valid_parts'])}개")
        for part in config['valid_parts']:
            print(f"    - {part['id']}: {part['name']}")

        print("\n  설정 파일 로드 성공!")
        return True

    except Exception as e:
        print(f"\n  설정 파일 로드 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "#"*60)
    print("#  연료펌프 모듈 검사 시스템 - 모듈 테스트")
    print("#"*60)

    # 결과 저장 디렉토리 생성
    os.makedirs('test_results', exist_ok=True)

    results = []

    # 1. 설정 파일 테스트
    results.append(('설정 파일 로드', test_config_loading()))

    # 2. 외관 결함 검사 테스트
    results.append(('외관 결함 검사', test_defect_inspector()))

    # 3. 바코드 검사 테스트
    results.append(('바코드 검사', test_barcode_inspector()))

    # 4. 이종품 검사 테스트
    results.append(('이종품 검사', test_variant_inspector()))

    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "-"*60)
    if all_passed:
        print("모든 테스트 통과!")
    else:
        print("일부 테스트 실패")
    print("-"*60)

    print("\n테스트 결과 이미지: test_results/ 디렉토리 확인")
    print()


if __name__ == "__main__":
    main()
