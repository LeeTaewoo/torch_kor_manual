require 'nn'
-- References: 
-- 1. Bishop, Neural Networks for Pattern Recognition, 1995, ch. 4.8-9.
-- 2. https://www.youtube.com/watch?v=-YRB0eFxeQA, 20:00~.
-- 3. http://roboticist.tistory.com/590
-- 4. https://groups.google.com/forum/#!searchin/torch7/requ/torch7/ZF5Z_9JmT8M/voSkymCdDgAJ

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  self.output[self.output:lt(0)] = 0
  self.output:cmul(self.output)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- Since z = x^2 if x>0 else z=0, the dz_dx = 2x if x>0 else dz_dx=0.
  -- gradInput = gradOutput * dz_dx
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  local dz_dx = torch.gt(input,0):type('torch.DoubleTensor'):mul(2):cmul(input)
  self.gradInput:cmul(dz_dx)
  return self.gradInput
end

