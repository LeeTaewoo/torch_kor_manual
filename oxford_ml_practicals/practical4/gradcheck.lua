local create_model = require 'create_model'

--------------------------------------------------------------
-- 설정
local opt = { nonlinearity_type = 'requ' }
--local opt = { nonlinearity_type = 'sigmoid' }

-- 손실의 기울기를 수치적으로 확인하는 함수.
-- f는 스칼라 값으로 된 함수 (오차를 계산), E(x;w).
-- g는 진짜 기울기를 리턴합니다 (f로 들어가는 입력은 1차원 텐서라고 가정합니다), dE/dw.
-- 차이, 진짜 기울기, 그리고 추정된 기울기를 리턴합니다.
local function checkgrad(f, g, x, eps)
  -- 진짜 기울기 계산
  local grad = g(x)
  
  -- 기울기의 수치적 근사치 계산
  local eps = eps or 1e-7
  local two_eps = eps + eps
  local grad_est = torch.DoubleTensor(grad:size())

  for i = 1, grad:size(1) do
    x[i] = x[i] + eps
    grad_est[i]= f(x)
    x[i] = x[i] - two_eps
    grad_est[i]= ( grad_est[i] - f(x) ) / two_eps
    x[i] = x[i] + eps
  end

  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  return diff, grad, grad_est
end

function fakedata(n)
    local data = {}
    data.inputs = torch.randn(n, 4)                     
    data.targets = torch.rand(n):mul(3):add(1):floor()  
    return data
end

---------------------------------------------------------
-- 가짜 데이터를 만들어, 기울기 확인 수행
--
torch.manualSeed(1)
local data = fakedata(5)
local model, criterion = create_model(opt)
local parameters, gradParameters = model:getParameters()

-- loss(params) 리턴
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  return criterion:forward(model:forward(data.inputs), data.targets)
end

-- dloss(params)/dparams 리턴
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local outputs = model:forward(data.inputs)
  criterion:forward(outputs, data.targets)
  model:backward(data.inputs, criterion:backward(outputs, data.targets))

  return gradParameters
end

local err = checkgrad(f, g, parameters)
local precision= 1e-4
print('precision: ' .. precision)
print('==> error:' .. err)
if err<precision then
  print('==> module OK')
else
  print('==> error too large, incorrect implementation')
end
