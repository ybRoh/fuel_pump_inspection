"""
연료펌프 모듈 검사 모듈
Fuel Pump Module Inspection Modules
"""

from .defect_inspector import DefectInspector, DefectType, Defect
from .barcode_inspector import BarcodeInspector, BarcodeResult
from .variant_inspector import VariantInspector, VariantResult
from .deep_learning_inspector import DeepLearningInspector, DLInspectionResult, Detection

__all__ = [
    # 기존 검사 모듈
    'DefectInspector',
    'DefectType',
    'Defect',
    'BarcodeInspector',
    'BarcodeResult',
    'VariantInspector',
    'VariantResult',
    # 딥러닝 검사 모듈
    'DeepLearningInspector',
    'DLInspectionResult',
    'Detection',
]
