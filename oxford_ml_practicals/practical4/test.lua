require 'nn'
require 'requ'

module= nn.Sequential()
module:add(nn.ReQU())
--module:add(nn.Sigmoid())

------------------------------------------
-- ReQU forward Test
--
input=torch.randn(10)
print(input)
output=module:forward(input)
print(output)

--[[
-- module:forward output
for i= 1, output:size(1) do
  print(output[i])
end
--]]

-- Manual ReQU forward output
for i= 1, input:size(1) do
  if input[i]<0 then
    input[i]=0
  else 
    input[i]= input[i]*input[i]
  end
  print(input[i])  
end

------------------------------------------
-- Jacobian Test
--
local precision= 1e-5
local jac= nn.Jacobian
local err= jac.testJacobian(module,input)

print('==> error:' .. err)
if err<precision then
  print('==> module OK')
else
  print('==> error too large, incorrect implementation')
end

