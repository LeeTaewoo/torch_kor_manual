require 'torch'
require 'math'
local loader = require 'iris_loader'
local train = require 'train'

torch.manualSeed(1)
local data = loader.load_data()

local opt = {
  nonlinearity_type = 'sigmoid',
  training_iterations = 150, -- 노트: 이 코드는 *minibatch*가 아닌 *배치(batche)*를 사용합니다.
  print_every = 25,          -- 몇 번의 반복마다 손실을 출력할지
}

-- sigmoid와 requ 버전을 훈련
model_sigmoid, losses_sigmoid = train(opt, data)
-- TODO: requ를 구현하였으면 주석을 해제하십시오.
opt.nonlinearity_type = 'requ'
model_requ, losses_requ = train(opt, data)


---[[
--------------------------------------------------------
-- 평가: 이 다음부터의 모든 코드는 무시될 수 있습니다.
-- 노트: 비록 우리에게는 시험 세트가 없지만, 우리는 두 훈련 손실 곡선을 그릴 것입니다.
-- 우리는 모델이 과적응되었는지 알 수 없을 것입니다. 
-- 그러나 우리는 우리의 모델이 얼마나 유연하지 볼 수 있습니다.

-- 그리기
gnuplot.figure()
gnuplot.plot({'sigmoid',
  torch.range(1, #losses_sigmoid), -- x-좌표
  torch.Tensor(losses_sigmoid),    -- y-좌표
  '-'}
  -- TODO: requ를 구현하였으면 주석을 해제하십시오.
   , {'requ',
   torch.range(1, #losses_requ),    -- x-좌표
   torch.Tensor(losses_requ),       -- y-좌표
   '-'}
  )

models = { 
    requ = model_requ,  -- TODO: requ를 구현하였으면 주석을 해제하십시오.
    sigmoid = model_sigmoid 
}
for model_name, model in pairs(models) do
  -- 훈련 세트에 대한 분류 오차
  local log_probs = model:forward(data.inputs)
  local _, predictions = torch.max(log_probs, 2)
  print(string.format('# correct for %s:', model_name))
  print(torch.mean(torch.eq(predictions:long(), data.targets:long()):double()))

  -- 한 슬라이스 안에서의 분류 영역 (pdf 문서의 그림 1 scatterplots를 참고하십시오)
  -- 아름답지는 않지만, gnuplot 코드를 크게 바꾸거나 다른 라이브러리를 쓰지 않고 얻을 수 있는 최선의 결과입니다.
  local f1 = 4 -- 첫 번째 축 위의 특징
  local f2 = 3 -- 두 번째 축 위의 특징
  local size = 60  -- 해상도
  local f1grid = torch.linspace(data.inputs[{{},f1}]:min(), data.inputs[{{},f1}]:max(), size)
  local f2grid = torch.linspace(data.inputs[{{},f2}]:min(), data.inputs[{{},f2}]:max(), size)
  local result = torch.Tensor(size, size)
  local input = data.inputs[1]:clone()
  for i=1,size do
    input[f1] = f1grid[i]
    for j=1,size do
      input[f2] = f2grid[j]
      result[{i,j}] = math.exp(model:forward(input)[1])
    end
  end
  result[1][1] = 0 -- 올바른 스케일을 얻귀 위한 코드 수정(ugly hack)
  result[1][2] = 1 -- 올바른 스케일을 얻귀 위한 코드 수정(ugly hack)
  gnuplot.figure()
  gnuplot.imagesc(result, model_name)
end
--]]
