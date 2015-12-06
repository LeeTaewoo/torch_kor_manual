require 'nngraph'

m = nn.Sequential()
m:add(nn.SplitTable(1))
m:add(nn.ParallelTable():add(nn.Linear(10, 20)):add(nn.Linear(10, 30)))
input = nn.Identity()()
input1, input2 = m(input):split(2)
m3 = nn.JoinTable(1)({input1, input2})

g = nn.gModule({input}, {m3})

indata = torch.rand(2, 10)
gdata = torch.rand(50)
g:forward(indata)
g:backward(indata, gdata)

graph.dot(g.fg, 'Forward Graph', 'Forward Graph')
graph.dot(g.bg, 'Backward Graph', 'Backward Graph')

