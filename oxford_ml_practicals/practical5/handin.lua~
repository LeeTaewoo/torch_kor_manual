require 'nngraph'

h1= nn.Identity()()
h2= nn.Identity()()
h3= nn.Identity()()
hh3= nn.Linear(3,5)(h3):annotate{name='hh3'}
hh2= nn.CMulTable()({h2, hh3})
z= nn.CAddTable()({h1, hh2})

g= nn.gModule({h1,h2,h3},{z})

x1= torch.randn(5)
x2= torch.randn(5)
x3= torch.randn(3)

print(x1)
print(x2)
print(x3)
print(g:forward({x1,x2,x3}))
graph.dot(g.fg, 'Forward Graph', 'Forward Graph_handin')

-------------------------------
-- Manual Test
--
local weight
local bias
for indexNode, node in ipairs(g.forwardnodes) do
  if node.data.annotations.name == "hh3" then 
    weight = node.data.module.weight 
    bias = node.data.module.bias
  end
end
--print(weight)
--print(bias)

-- z= x1 + ( x2 .* linear(x3) )
z1= weight * x3 + bias
z2= torch.cmul(x2,z1)
print(torch.add(x1,z2))

