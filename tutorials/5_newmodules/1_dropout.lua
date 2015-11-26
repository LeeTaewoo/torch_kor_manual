
-- 이 파일에서, 우리는 우리가 정의한 드롭아웃 모듈을 시험합니다:
require 'nn'
require 'DropoutEx'
require 'image'

-- 드롭아웃 객체 정의:
n = nn.DropoutEx(0.5)

-- 영상 로드:
i = image.lena()

-- 그 영상을 처리:
result = n:forward(i)

-- 결과 출력:
print('original image:')
itorch.image(i)
print('result image:')
itorch.image(result)

-- 약간의 통계:
mse = i:dist(result)
print('mse between original imgae and dropout-processed image: ' .. mse)
