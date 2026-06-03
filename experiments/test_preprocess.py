# -*- coding: utf-8 -*-
"""ทดสอบเปรียบเทียบ output ของทุก preprocessing strategy กับข้อความตัวอย่าง"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rris import utils

test_texts = [
    "อร่อยมากกกกก 😋😋 ราคา 350 บาท!!! Very good food https://example.com",
    "ไม่อร่อยเลย แย่มากๆๆๆ 1/10 จะไม่กลับมาอีกแล้ว NOT recommended!!!",
    "So so naja 3 stars na krab ก็โอเคอยู่นะ 🤷‍♂️",
]

for text in test_texts:
    print(f"\nORIGINAL: {text}")
    print("-" * 70)
    for name, fn in utils.PREPROCESS_REGISTRY.items():
        result = fn(text)
        print(f"  {name:15s}: {result}")
    print()
