
require 'nn'
require 'DropoutEx'

print '==> Jacobian으로 역전파 시험하기 (유한 요소)'

-- 모듈을 시험하기 위해, 우리는 랜덤성을 고정시킬 필요가 있습니다,
-- Jacobian 시험기가 모듈의 출력이 결정적(deterministic)일 것으로
-- 기대하기 때문입니다...
-- 그래서 우리가 전체 시험을 위해 랜덤 잡음을 한 번만 만든다는 점을
-- 제외하면, 코드는 같습니다.
firsttime = true
function nn.DropoutEx.updateOutput(self, input)
   if firsttime then
      self.noise:resizeAs(input)
      if self.p > 0 then
         self.noise:bernoulli(1-self.p)
      else
         self.noise:zero()
      end
      firsttime = false
   end
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   self.output:div(1-self.p)
   return self.output
end

-- 파라미터들
local precision = 1e-5
local jac = nn.Jacobian

-- 입력과 모듈을 정의
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 0.25
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.DropoutEx(percentage)

-- Jacobian으로 역전파를 시험
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
