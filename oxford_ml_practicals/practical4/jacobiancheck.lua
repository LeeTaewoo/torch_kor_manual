require 'requ'

local function jacobian_wrt_input(module, x, eps)
  local z = module:forward(x):clone()                 
  local jac = torch.DoubleTensor(z:size(1), x:size(1))

  local one_hot = torch.zeros(z:size())
  for i = 1, z:size(1) do
    one_hot[i] = 1 
    jac[i]:copy(module:backward(x, one_hot))
    one_hot[i] = 0
  end

  local jac_est = torch.DoubleTensor(z:size(1), x:size(1))

--[[ One-sided test
  for i = 1, x:size(1) do
    x[i] = x[i] + eps
    local z_offset = module:forward(x)
    x[i] = x[i] - eps
    jac_est[{{},i}]:copy(z_offset):add(-1, z):div(eps)
  end
--]]

---[[ Two-sided test
  for i = 1, x:size(1) do
    x[i] = x[i] + eps
    local z_offset = module:forward(x):clone()
    x[i] = x[i] - 2*eps
    local z_offset2 = module:forward(x)
    x[i] = x[i] + eps
    jac_est[{{},i}]:copy(z_offset):add(-1, z_offset2):div(2*eps)
  end
--]]

  local abs_diff = (jac - jac_est):abs()
  return jac, jac_est, torch.mean(abs_diff), torch.min(abs_diff), torch.max(abs_diff)
end

torch.manualSeed(1)
local module = nn.ReQU()
--local module= nn.Sigmoid()

local x = torch.randn(10)
jac, jac_est, err_mean, err_min, err_max= jacobian_wrt_input(module, x, 1e-6)
print('True Jacobian matrix')
print(jac)
print('Estimated Jacobian matrix')
print(jac_est)
local precision= 1e-6
local err= err_max
print('precision: ' .. precision)
print('==> error:' .. err)
if err<precision then
  print('==> module OK')
else
  print('==> error too large, incorrect implementation')
end
