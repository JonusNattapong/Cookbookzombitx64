"""
Cookbookzombitx64: ไลบรารี่สำหรับการพัฒนาโมเดลเอไอที่ทุกคนเข้าถึงได้ง่าย
==========================================================================

ไลบรารี่นี้มุ่งเน้นการพัฒนาโมเดลเอไอที่เข้าถึงง่าย และมีฟังก์ชันการทำงานมากมาย
ตั้งแต่การโหลดและบันทึกโมเดล ไปจนถึงการปรับแต่งไฮเปอร์พารามิเตอร์
"""

__version__ = "0.1.0"

# นำเข้าโมดูลต่างๆ 
from . import llm
from . import vlm
from . import sentence_transformers
from . import text_tasks
from . import image_tasks
from . import tabular_tasks
from . import data
from . import training
from . import utils

# ฟังก์ชันช่วยเหลือสำหรับการเริ่มต้นใช้งาน
def version():
    """แสดงเวอร์ชันของไลบรารี่"""
    return __version__

def help():
    """แสดงข้อมูลช่วยเหลือการใช้งานไลบรารี่"""
    print("Cookbookzombitx64: ไลบรารี่สำหรับการพัฒนาโมเดลเอไอที่ทุกคนเข้าถึงได้ง่าย")
    print("เวอร์ชัน:", __version__)
    print("\nโมดูลที่มี:")
    print("- llm: การปรับแต่งโมเดลภาษาขนาดใหญ่")
    print("- vlm: การปรับแต่งโมเดลภาพและภาษา")
    print("- sentence_transformers: การทำงานกับ Sentence Transformers")
    print("- text_tasks: งานประมวลผลข้อความอื่นๆ")
    print("- image_tasks: งานประมวลผลภาพ")
    print("- tabular_tasks: งานประมวลผลข้อมูลตาราง")
    print("- data: การจัดการชุดข้อมูล")
    print("- training: การฝึกโมเดล")
    print("- utils: ฟังก์ชันช่วยเหลือต่างๆ")
    print("\nดูตัวอย่างได้ที่: examples/") 