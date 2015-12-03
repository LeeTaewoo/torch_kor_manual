# 신경망 그래프 패키지

[![Build Status](https://travis-ci.org/torch/nngraph.svg)](https://travis-ci.org/torch/nngraph) 

원문: <https://github.com/torch/nngraph/>

이 패키지는 [Torch](https://github.com/torch/torch7/blob/master/README.md)의 `nn` 라이브러리를 위한 그래픽적 계산을 제공합니다.

## 요구사항

이 라이브러리를 사용하기 위해 `graphviz`가 필요한 것은 아닙니다. 그러나 만약 당신이 `graphviz`를 가지고 있다면, 당신이 만든 그래프들을 출력할 수 있을 것입니다. 그 패키지를 설치하기 위해, 아래 명령어를 실행시키십시오:

```bash
# 맥 사용자
brew install graphviz
# 데비안/우분투 사용자
sudo apt-get install graphviz -y
```

## 사용

[플러그: 옥스퍼드의 난도 드 페레이라가 더 친절하게 설명한 nngraph 튜토리얼](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf)

이 라이브러리의 목표는 `nn` 패키지의 사용자에게 복잡한 구조들을 쉽게 만들 수 있는 도구들을 제공하는 것입니다.
어떤 주어진 `nn` `module`도 *그래프 노드* 하나로 들어갑니다.
`nn.Module`의 인스턴스의 `__call__` 연산자는 마치 함수 호출들을 작성하고 있는 것처럼 구조들을 만드는 데 사용됩니다.  

### 두 개의 숨겨진 층을 가진 다층 퍼셉트론

```lua
h1 = nn.Linear(20, 10)()
h2 = nn.Linear(10, 1)(nn.Tanh()(nn.Linear(10, 10)(nn.Tanh()(h1))))
mlp = nn.gModule({h1}, {h2})

x = torch.rand(20)
dx = torch.rand(1)
mlp:updateOutput(x)
mlp:updateGradInput(x, dx)
mlp:accGradParameters(x, dx)

-- 그래프 그리기 (포워드 그래프, '.fg')
graph.dot(mlp.fg, 'MLP')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp.png" width="300px"/>

이 다이어그램을 꼭대기부터 바닥까지 읽으십시오. 첫 번째와 마지막 노드는 *더미(가짜) 노드*입니다.
*더미 노드*는 그 그래프의 모든 입력들과 출력들을 다시 그룹으로 묶습니다.  
`module` 엔트리는 그 노드의 기능을 설명합니다. 
`module` 엔트리는 `input`에 적용되고 그 모양 `gradOutput`의 결과를 생성합니다.
`mapindex`는 부모 노드들을 가리키는 포인터들을 담고 있습니다.

그 *그래프*를 파일로 저장하기 위해, 그 파일 이름을 특정합니다, 그리고 `dot` 하나와 `svg` 파일들이 저장될 것입니다.
예를 들어, 다음과 같이 쓸 수 있습니다: 

```lua
graph.dot(mlp.fg, 'MLP', 'myMLP')
```


### 두 입력과 두 출력을  가진 네트워크

```lua
h1 = nn.Linear(20, 20)()
h2 = nn.Linear(10, 10)()
hh1 = nn.Linear(20, 1)(nn.Tanh()(h1))
hh2 = nn.Linear(10, 1)(nn.Tanh()(h2))
madd = nn.CAddTable()({hh1, hh2})
oA = nn.Sigmoid()(madd)
oB = nn.Tanh()(madd)
gmod = nn.gModule({h1, h2}, {oA, oB})

x1 = torch.rand(20)
x2 = torch.rand(10)

gmod:updateOutput({x1, x2})
gmod:updateGradInput({x1, x2}, {torch.rand(1), torch.rand(1)})
graph.dot(gmod.fg, 'Big MLP')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp2.png" width="300px"/>


### 컨테이너들을 가진 네트워크

출력들로 구성된 테이블 하나를 출력하는 (`ParallelTable` 같은) 컨테이너 모듈들을 사용하는 또다른 네트워크.

```lua
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

graph.dot(g.fg, 'Forward Graph')
graph.dot(g.bg, 'Backward Graph')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp3_forward.png" width="300px"/>
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp3_backward.png" width="300px"/>


### 더 재미있는 그래프들

각 층이 이전 두 층의 출력을 입력으로 받는 다층 네트워크.

```lua
input = nn.Identity()()
L1 = nn.Tanh()(nn.Linear(10, 20)(input))
L2 = nn.Tanh()(nn.Linear(30, 60)(nn.JoinTable(1)({input, L1})))
L3 = nn.Tanh()(nn.Linear(80, 160)(nn.JoinTable(1)({L1, L2})))

g = nn.gModule({input}, {L3})

indata = torch.rand(10)
gdata = torch.rand(160)
g:forward(indata)
g:backward(indata, gdata)

graph.dot(g.fg, 'Forward Graph')
graph.dot(g.bg, 'Backward Graph')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp4_forward.png" width="300px"/>
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp4_backward.png" width="300px"/>


## 주석

네트워크에 주석을 다는 것도 가능합니다. 노드들에 이름 또는 속성을 레이블링할 수 있습니다.
그 레이블들은 우리가 네트워크를 그래프로 그릴 때 나타납니다.
이것은 거대한 그래프들을 다룰 때 도움이 될 수 있습니다.

그래프 속성의 완전한 목록은 [graphviz 문서](http://www.graphviz.org/doc/info/attrs.html)를 보십시오.

```lua
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

graph.dot(g.fg, 'Forward Graph', '/tmp/fg')
graph.dot(g.bg, 'Backward Graph', '/tmp/bg')
```

이 경우에, 그래프들은 다음 네 개의 파일들로 저장됩니다: `/tmp/{fg,bg}.{dot,svg}`.

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/annotation_fg.png" width="300px"/>
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/annotation_bg.png" width="300px"/>

## 디버깅

nngraph로, 우리는 매우 복잡한 네트워크들을 만들 수 있습니다.
이런 경우, 우리는 오류를 찾기가 어려울 수 있습니다.
그 목적으로, nngraph는 몇 가지 유용한 유틸리티들을 제공합니다.
다음 짧은 코드는 한 그래프에서 노드들에 주석을 달기 위해 어떻게 지역 변수 이름들을 사용하는지를 보여줍니다.
그리고 어떻게 디버깅 모드를 활성화하는지를 보여줍니다.
디버깅 모드는 런타임 에러의 경우가 써 넣어진 에러 노드와 함께 자동으로 svg 파일 하나를 만듭니다.

```lua

require 'nngraph'

-- 문제 노드가 강조된 그 그래프의 SVG 생성합니다.  
-- 그리고 svg에서 노드들에 마우스 포인터를 올려놓으면, filename:line_number 정보를 볼 수 있습니다.
-- 디버그 모드가 활성화 되지 않더라도, 노드들은 지역 변수 이름으로 주석이 달릴 것입니다.  
nngraph.setDebug(true)

local function get_net(from, to)
	local from = from or 10
	local to = to or 10
	local input_x = nn.Identity()()
	local linear_module = nn.Linear(from, to)(input_x)

	-- 노드들에 지역 변수 이름의 주석을 답니다.
	nngraph.annotateNodes()
	return nn.gModule({input_x},{linear_module})
end

local net = get_net(10,10)

-- 만약 당신이 그 네트워크에 이름을 주면, 에러가 생긴 경우 svg에 그 이름을 사용합니다.
-- 만약 당신이 그 네트워크에 이름을 주지 않으면, 이름은 그 그래프의 입력과 출력 개수로 임의로 만들어 사용됩니다.
net.name = 'my_bad_linear_net'

-- 에러가 생기도록 일부러 틀린 차원을 가진 입력 하나를 준비합니다.
local input = torch.rand(11)
pcall(function() net:updateOutput(input) end)
-- 이 명령은 에러를 만들고 그래프 하나를 출력합니다.
```
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/my_bad_linear_net.png" width="300px"/>

