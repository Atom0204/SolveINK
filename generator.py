from trdg.generators import GeneratorFromStrings
import os
import random

# Function to generate arithmetic expressions
def generate_arithmetic_expressions(n):
    ops = ['+', '-', '*', '/']
    expressions = []

    for _ in range(n):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        c = random.randint(1, 9)
        op1 = random.choice(ops)
        op2 = random.choice(ops)
        use_parens = random.choice([True, False])
        use_float = random.choice([True, False])

        if use_float:
            a = round(random.uniform(1, 9), 1)
            b = round(random.uniform(1, 9), 1)

        expr = f'({a}{op1}{b}){op2}{c}' if use_parens else f'{a}{op1}{b}{op2}{c}'

        if random.choice([True, False]):
            expr = '-' + expr

        if random.choice([True, False]):
            expr += f'**{random.randint(1, 2)}'

        expressions.append(expr)
    return expressions

# Combine expressions
static_expressions = ['2+3', '5*(2+1)', '7-4', '8/2', '3+5*2']
arithmetic_expressions = generate_arithmetic_expressions(95)
all_expressions = static_expressions + arithmetic_expressions

# Generator
generator = GeneratorFromStrings(
    all_expressions,
    count=len(all_expressions),
    size=60,
    skewing_angle=10,
    random_skew=True,
    blur=1,
    random_blur=True,
    background_type=0
)

# Output directories
os.makedirs('math_ocr_dataset/images', exist_ok=True)
os.makedirs('math_ocr_dataset/annotations', exist_ok=True)

# Generate and save images/labels
for i, (img, lbl) in enumerate(generator):
    img.save(f'math_ocr_dataset/images/synthetic_{i:03d}.jpg')
    with open(f'math_ocr_dataset/annotations/synthetic_{i:03d}.txt', 'w') as f:
        f.write(lbl)
