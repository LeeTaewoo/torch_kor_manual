require 'nngraph'
-- Run: th ex1.lua
-- Refernce: https://groups.google.com/forum/?fromgroups#!topic/torch7/7uoKNv_InVA

h1 = nn.Linear(20, 10)()
h2 = nn.Linear(10, 1)( nn.Tanh()( nn.Linear(10,10)(nn.Tanh()(h1))))
--   nn.Linear(10, 1)(                                            )
--                     nn.Tanh()(                                )
--                                nn.Linear(10,10)(             )
--                                                 nn.Tanh()(h1)
mlp = nn.gModule({h1}, {h2})

x = torch.rand(20)
dx = torch.rand(1)
mlp:updateOutput(x)
mlp:updateGradInput(x, dx)
mlp:accGradParameters(x, dx)


function print_outputs(m)
  print(m)
  print(m.output)
end
mlp:apply(print_outputs)

--graph.dot(mlp.fg, 'MLP')
graph.dot(mlp.fg, 'MLP', 'MLP1')
