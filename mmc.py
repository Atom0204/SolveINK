from trdg.generators import GeneratorFromStrings
import os

expressions = [
    '0+1', '2+3', '4+5', '6+7', '8+9',
    '9-8', '7-6', '5-4', '3-2', '1-0',
    '1*2', '3*4', '5*6', '7*8', '9*0',
    '2/1', '4/2', '6/3', '8/4', '9/3',
    '1**2', '2**3', '3**2', '4**1', '5**0',
    '(1+2)', '(3*4)', '(5-6)', '(7/8)', '(9+0)',
    '1.1+2.2', '3.3-1.1', '4.4*0.5', '6.6/3.3', '5.0**1',
    '-1+2', '-3*4', '-(5+6)', '-7.0+2', '-(8/2)',
    '((1+2)*3)', '((4-1)/3)', '(2+(3*4))-5', '6-(2+1)', '7+(8/4)-1',
    '3*(4+5)', '(6/2)+(1*3)', '(9-3)*(2+1)', '2+(3*4)-(1/2)', '4**(1+1)',
    '0+0', '1+1+1', '2*2*2', '3+(3-3)', '4+(5*(6-2))',
    '(7+(8-1))', '9/(3+0)', '((1+1)+2)**2', '3**(2-1)', '(4+4)-(2*2)',
    '6+(3*3)-3', '7-(2+2)+1', '(8-4)*2', '9-(1*1)', '10-(4+3)',
    '5+(2.5*3)', '1.1+2.2-3.3', '2+(3.5/1.4)', '3*(2.0+1.0)', '4.5-1.5+2'
]

generator = GeneratorFromStrings(
    expressions,
    count=100,
    size=60,  # Font size
    skewing_angle=10,
    random_skew=True,
    blur=1,
    random_blur=True,
    background_type=0  # White background
)

os.makedirs('math_ocr_dataset/images', exist_ok=True)
os.makedirs('math_ocr_dataset/annotations', exist_ok=True)

for i, (img, lbl) in enumerate(generator):
    img.save(f'math_ocr_dataset/images/synthetic_{i:03d}.jpg')
    with open(f'math_ocr_dataset/annotations/synthetic_{i:03d}.txt', 'w') as f:
        f.write(lbl)