require 'nngraph'

input = nn.Identity()()
L1 = nn.Tanh()(nn.Linear(10, 20)(input)):annotate{
   name = 'L1', description = 'Level 1 Node',
   graphAttributes = {color = 'red'}
}
L2 = nn.Tanh()(nn.Linear(30, 60)(nn.JoinTable(1)({input, L1}))):annotate{
   name = 'L2', description = 'Level 2 Node',
   graphAttributes = {color = 'blue', fontcolor = 'green'}
}
L3 = nn.Tanh()(nn.Linear(80, 160)(nn.JoinTable(1)({L1, L2}))):annotate{
   name = 'L3', descrption = 'Level 3 Node',
   graphAttributes = {color = 'green',
   style = 'filled', fillcolor = 'yellow'}
}

g = nn.gModule({input},{L3})

indata = torch.rand(10)
gdata = torch.rand(160)
g:forward(indata)
g:backward(indata, gdata)

graph.dot(g.fg, 'Forward Graph', 'fg_ex5')
graph.dot(g.bg, 'Backward Graph', 'bg_ex5')
