from pathlib import Path
import platform  # portable
import re

print("platform.node():", platform.node())

COLAB = not not re.findall(r'[a-z\d]{12}', platform.node())

# assert COLAB

DIRNAME = '/content' if COLAB else '.'

if COLAB:
    if not Path('data').exists():
        !git clone https://github.com/yucongo/data.git
    else:
        os.chdir('data')
        !git pull
    os.chdir(DIRNAME)
