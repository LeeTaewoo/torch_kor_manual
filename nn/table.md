<a name="nn.TableLayers"></a>
# 테이블 층 #

이 일련의 모듈들은 신경망의 층들에서 `테이블`들을 조작할 수 있게 합니다.
이는 우리가 매우 풍부한 구조들을 만들 수 있게 합니다:

  * `테이블` 컨테이너 모듈들은 하위 모듈들을 캡슐화합니다:
    * [`ConcatTable`](#nn.ConcatTable): 각 멤버 모듈을 같은 입력 [`Tensor`](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor)에 적용하고, `테이블` 하나를 출력합니다 ;
    * [`ParallelTable`](#nn.ParallelTable): `i` 번째 멤버 모듈을 `i` 번째 입력에 적용하고, `테이블` 하나를 출력합니다 ;
  * 테이블 전환(conversion) 모듈들은 `테이블`과 `텐서` 또는 `테이블` 사이를 전환시킵니다:
    * [`SplitTable`](#nn.SplitTable): `텐서` 하나를 `텐서`들로 구성된 `테이블` 하나로 나눕니다 ;
    * [`JoinTable`](#nn.JoinTable): `텐서`들로 구성된 `테이블` 하나를 `텐서` 하나로 합칩니다 ;
    * [`MixtureTable`](#nn.MixtureTable): 전문가들의 혼합물(mixture of experts)은 여러 인공 신경망(전문가)의 예측들을 게이터(gater) 하나로 모아 한 과제를 수행하는 전략입니다. `MixtureTable`은 게이터(gater)에 의해 가중된 전문가들의 혼합물입니다 ;
    * [`SelectTable`](#nn.SelectTable): `테이블`에서 요소 하나를 선택합니다 ;
    * [`NarrowTable`](#nn.NarrowTable): `테이블`에서 슬라이스 하나를 선택합니다 ;
    * [`FlattenTable`](#nn.FlattenTable): 중첩된 `테이블` 구조를 평평하게 만듭니다(flatten) ;
  * 입력 `텐서`들의 (`테이블`) 쌍에서, 페어 모듈들은 거리 또는 유사도 같은 척도를 계산합니다:
    * [`PairwiseDistance`](#nn.PairwiseDistance): `p`-놈을 출력합니다. 입력들 사이 거리 ;
    * [`DotProduct`](#nn.DotProduct): 입력들 사이 점곱(dot product, 유사도)을 출력합니다 ;
    * [`CosineDistance`](#nn.CosineDistance): 입력들 사이 코사인 거리를 출력합니다 ;
  * CMath 모듈들ㅇㄴ 요소별 연산들을 `텐서`들의 `테이블`에 수행합니다:
    * [`CAddTable`](#nn.CAddTable): 입력 `텐서`들의 덧셈;
    * [`CSubTable`](#nn.CSubTable): 입력 `텐서`들의 뺄셈;
    * [`CMulTable`](#nn.CMulTable): 입력 `텐서`들의 곱셈;
    * [`CDivTable`](#nn.CDivTable): 입력 `텐서`들의 나눗셈;
  * 오차 판정 기준(criterion)들의 `테이블`:
    * [`CriterionTable`](#nn.CriterionTable): 입력들로 구성된 `테이블`을 받을 수 있도록 한 [Criterion](criterion.md#nn.Criterion)의 코드를 수정합니다.

`테이블` 기반 모듈들에서 `forward()`와 `backward()` 메소드는 입력들로 `테이블`들을 받을 수 있습니다.
보통 [`Sequential`](containers.md#nn.Sequential) 모듈도 이것을 할 수 있습니다.
그래서 우리에게 필요한 모든 것은 그런 `테이블`들을 이용하는 다른 자식 모듈들입니다 

```lua
mlp = nn.Sequential()
t = {x, y, z}
pred = mlp:forward(t)
pred = mlp:forward{x, y, z}      -- 이것은 바로 앞 줄과 같습니다
```

<a name="nn.ConcatTable"></a>
## ConcatTable ##

```lua
module = nn.ConcatTable()
```

`ConcatTable`은 각 멤버 모듈에 같은 입력
[`텐서`](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor) 또는 `테이블`을 
적용하는 컨테이너 모듈입니다.

```
                  +----------+
             +----> {멤버1,  |
+-------+    |    |          |
| 입력  +----+---->  멤버2,  |
+-------+    |    |          |
  또는       +---->  멤버3}  |
 {입력}           +----------+
```

### 예 1

```lua
mlp = nn.ConcatTable()
mlp:add(nn.Linear(5, 2))
mlp:add(nn.Linear(5, 3))

pred = mlp:forward(torch.randn(5))
for i, k in ipairs(pred) do print(i, k) end
```

는 다음을 출력합니다:

```lua
1
-0.4073
 0.0110
[torch.Tensor of dimension 2]

2
 0.0027
-0.0598
-0.1189
[torch.Tensor of dimension 3]
```

### 예 2

```lua
mlp = nn.ConcatTable()
mlp:add(nn.Identity())
mlp:add(nn.Identity())

pred = mlp:forward{torch.randn(2), {torch.randn(3)}}
print(pred)
```

는 다음을 출력합니다 ([th](https://github.com/torch/trepl)를 사용하여):

```lua
{
  1 :
    {
      1 : DoubleTensor - size: 2
      2 :
        {
          1 : DoubleTensor - size: 3
        }
    }
  2 :
    {
      1 : DoubleTensor - size: 2
      2 :
        {
          1 : DoubleTensor - size: 3
        }
    }
}
```


<a name="nn.ParallelTable"></a>
## ParallelTable ##

```lua
module = nn.ParallelTable()
```

`ParallelTable`은 그것의 `forward()` 메소드에서 `i` 번째 멤버 모듈을 `i` 번째 입력으로 
적용하고, 출력들의 집합으로 구성된 `테이블` 하나를 출력하는 
컨테이너 모듈입니다.


```
+----------+         +----------+
| {입력1,  +---------> {멤버1,  |
|          |         |          |
|  입력2,  +--------->  멤버2,  |
|          |         |          |
|  입력3}  +--------->  멤버3}  |
+----------+         +----------+
```

### 예

```lua
mlp = nn.ParallelTable()
mlp:add(nn.Linear(10, 2))
mlp:add(nn.Linear(5, 3))

x = torch.randn(10)
y = torch.rand(5)

pred = mlp:forward{x, y}
for i, k in pairs(pred) do print(i, k) end
```

는 다음을 출력합니다:

```lua
1
 0.0331
 0.7003
[torch.Tensor of dimension 2]

2
 0.0677
-0.1657
-0.7383
[torch.Tensor of dimension 3]
```


<a name="nn.SplitTable"></a>
## SplitTable ##

```lua
module = SplitTable(dimension, nInputDims)
```

[`텐서`](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor) 하나를 입력으로 받고
여러 개의 `테이블`들을 출력하는 모듈 하나를 만듭니다.
그 출력하는 `테이블`들은 `텐서`를 특정된 차원 `dimension`을 따라 나눈 것입니다.
아래 다이어그램에서, `dimension`은 `1`로 설정되었습니다.

```
    +----------+         +----------+
    | 입력[1]  +---------> {멤버1,  |
  +----------+-+         |          |
  | 입력[2]  +----------->  멤버2,  |
+----------+-+           |          |
| 입력[3]  +------------->  멤버3}  |
+----------+             +----------+
```

선택적 파라미터 `nInputDims`는 이 모듈이 받을 차원 구성을 특정할 수 있게 합니다.
이것은 미니배치와 비-미니배치 `텐서` 모두를 같은 모듈로 `forward` 할 수 있게 합니다.


### 예 1

```lua
mlp = nn.SplitTable(2)
x = torch.randn(4, 3)
pred = mlp:forward(x)
for i, k in ipairs(pred) do print(i, k) end
```

는 다음을 출력합니다:

```lua
1
 1.3885
 1.3295
 0.4281
-1.0171
[torch.Tensor of dimension 4]

2
-1.1565
-0.8556
-1.0717
-0.8316
[torch.Tensor of dimension 4]

3
-1.3678
-0.1709
-0.0191
-2.5871
[torch.Tensor of dimension 4]
```

### 예 2

```lua
mlp = nn.SplitTable(1)
pred = mlp:forward(torch.randn(4, 3))
for i, k in ipairs(pred) do print(i, k) end
```

는 다음을 출력합니다:

```lua
1
 1.6114
 0.9038
 0.8419
[torch.Tensor of dimension 3]

2
 2.4742
 0.2208
 1.6043
[torch.Tensor of dimension 3]

3
 1.3415
 0.2984
 0.2260
[torch.Tensor of dimension 3]

4
 2.0889
 1.2309
 0.0983
[torch.Tensor of dimension 3]
```

### 예 3

```lua
mlp = nn.SplitTable(1, 2)
pred = mlp:forward(torch.randn(2, 4, 3))
for i, k in ipairs(pred) do print(i, k) end
pred = mlp:forward(torch.randn(4, 3))
for i, k in ipairs(pred) do print(i, k) end
```

는 다음을 출력합니다:

```lua
1
-1.3533  0.7448 -0.8818
-0.4521 -1.2463  0.0316
[torch.DoubleTensor of dimension 2x3]

2
 0.1130 -1.3904  1.4620
 0.6722  2.0910 -0.2466
[torch.DoubleTensor of dimension 2x3]

3
 0.4672 -1.2738  1.1559
 0.4664  0.0768  0.6243
[torch.DoubleTensor of dimension 2x3]

4
 0.4194  1.2991  0.2241
 2.9786 -0.6715  0.0393
[torch.DoubleTensor of dimension 2x3]


1
-1.8932
 0.0516
-0.6316
[torch.DoubleTensor of dimension 3]

2
-0.3397
-1.8881
-0.0977
[torch.DoubleTensor of dimension 3]

3
 0.0135
 1.2089
 0.5785
[torch.DoubleTensor of dimension 3]

4
-0.1758
-0.0776
-1.1013
[torch.DoubleTensor of dimension 3]
```

또한, 이 모듈은 네거티브 차원들을 사용하여 끝에서부터의 인덱싱도 지원합니다.
이것은 입력의 차원 구성을 모르더라도 그 모듈을 사용할 수 있게 합니다.

### 예

```lua
m = nn.SplitTable(-2)
out = m:forward(torch.randn(3, 2))
for i, k in ipairs(out) do print(i, k) end
out = m:forward(torch.randn(1, 3, 2))
for i, k in ipairs(out) do print(i, k) end
```

는 다음을 출력합니다:

```
1
 0.1420
-0.5698
[torch.DoubleTensor of size 2]

2
 0.1663
 0.1197
[torch.DoubleTensor of size 2]

3
 0.4198
-1.1394
[torch.DoubleTensor of size 2]


1
-2.4941
-1.4541
[torch.DoubleTensor of size 1x2]

2
 0.4594
 1.1946
[torch.DoubleTensor of size 1x2]

3
-2.3322
-0.7383
[torch.DoubleTensor of size 1x2]
```

### 더 복잡한 예

```lua
mlp = nn.Sequential()       -- 텐서 하나를 입력으로 받는 네트워크 하나를 만듭니다.
mlp:add(nn.SplitTable(2))
c = nn.ParallelTable()      -- 두 텐서 슬라이스가 두 개의 다른 선형 층으로 병렬로 들어갑니다.
c:add(nn.Linear(10, 3))
c:add(nn.Linear(10, 7))
mlp:add(c)                  -- 두 요소를 가진 테이블을 출력합니다.
p = nn.ParallelTable()      -- 이 테이블들은 두 개의 추가 선형 층들로 따로따로 들어갑니다.
p:add(nn.Linear(3, 2))
p:add(nn.Linear(7, 1))
mlp:add(p)
mlp:add(nn.JoinTable(1))    -- 마침내, 테이블이 함께 합쳐지고 출력됩니다.

pred = mlp:forward(torch.randn(10, 2))
print(pred)

for i = 1, 100 do           -- 그런 네트워크를 훈련하는 몇 단계...
   x = torch.ones(10, 2)
   y = torch.Tensor(3)
   y:copy(x:select(2, 1, 1):narrow(1, 1, 3))
   pred = mlp:forward(x)

   criterion = nn.MSECriterion()
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(0.05)

   print(err)
end
```


<a name="nn.JoinTable"></a>
## JoinTable ##

```lua
module = JoinTable(dimension, nInputDims)
```

차원 `dimension`을 따라 그것들을 결합함으로써, 
`텐서`들로 구성된 `테이블` 하나를 입력으로 받고 `텐서` 하나를 출력하는 모듈을 만듭니다.
아래 다이어그램에서 `dimension`은 `1`로 설정되었습니다.

```
+----------+             +----------+
| {입력1,  +-------------> 출력[1]  |
|          |           +----------+-+
|  입력2,  +-----------> 출력[2]  |
|          |         +----------+-+
|  입력3}  +---------> 출력[3]  |
+----------+         +----------+
```

선택적 파라미터 `nInputDims`는 이 모듈이 받을 차원 구성을 특정할 수 있게 합니다.
이것은 미니배치와 비-미니배치 `텐서` 모두를 같은 모듈로 `forward` 할 수 있게 합니다.


### 예 1

```lua

y = torch.randn(5, 1)
z = torch.randn(2, 1)

print(nn.JoinTable(1):forward{x, y})
print(nn.JoinTable(2):forward{x, y})
print(nn.JoinTable(1):forward{x, z})
```

는 다음을 출력합니다:

```lua
 1.3965
 0.5146
-1.5244
-0.9540
 0.4256
 0.1575
 0.4491
 0.6580
 0.1784
-1.7362
[torch.DoubleTensor of dimension 10x1]

 1.3965  0.1575
 0.5146  0.4491
-1.5244  0.6580
-0.9540  0.1784
 0.4256 -1.7362
[torch.DoubleTensor of dimension 5x2]

 1.3965
 0.5146
-1.5244
-0.9540
 0.4256
-1.2660
 1.0869
[torch.Tensor of dimension 7x1]
```

### 예 2

```lua
module = nn.JoinTable(2, 2)

x = torch.randn(3, 1)
y = torch.randn(3, 1)

mx = torch.randn(2, 3, 1)
my = torch.randn(2, 3, 1)

print(module:forward{x, y})
print(module:forward{mx, my})
```

는 다음을 출력합니다:

```lua
 0.4288  1.2002
-1.4084 -0.7960
-0.2091  0.1852
[torch.DoubleTensor of dimension 3x2]

(1,.,.) =
  0.5561  0.1228
 -0.6792  0.1153
  0.0687  0.2955

(2,.,.) =
  2.5787  1.8185
 -0.9860  0.6756
  0.1989 -0.4327
[torch.DoubleTensor of dimension 2x3x2]
```

### 더 복잡한 예

```lua
mlp = nn.Sequential()         -- 텐서 하나를 입력으로 받는 네트워크 하나를 만듭니다.
c = nn.ConcatTable()          -- 그 같은 텐서가 두 개의 다른 선형 층으로 들어갑니다.
c:add(nn.Linear(10, 3))       
c:add(nn.Linear(10, 7))
mlp:add(c)                    -- 두 요소를 가진 테이블 하나를 출력합니다.
p = nn.ParallelTable()        -- 이 테이블들은 두 개의 추가 선형 층들로 따로따로 들어갑니다.
p:add(nn.Linear(3, 2))        
p:add(nn.Linear(7, 1))
mlp:add(p)
mlp:add(nn.JoinTable(1))      -- 마침내, 그 테이블들이 결합되어 출력됩니다.

pred = mlp:forward(torch.randn(10))
print(pred)

for i = 1, 100 do             -- 그런 네트워크를 훈련하는 몇 단계...
   x = torch.ones(10)
   y = torch.Tensor(3); y:copy(x:narrow(1, 1, 3))
   pred = mlp:forward(x)

   criterion= nn.MSECriterion()
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(0.05)

   print(err)
end
```


<a name='nn.MixtureTable'></a>
## MixtureTable ##

`module` = `MixtureTable([dim])`

`{게이터, 전문가들}` `테이블` 하나를 입력으로 받아 `게이터` `텐서`를 사용하여 
`전문가들`의 혼합물을 출력하는 모듈 하나를 만듭니다.
출력되는 `전문가들`의 혼합물은 `텐서` 하나 또는 `텐서`들로 구성된 테이블 하나입니다.
[전문가들의 혼합물(mixture of experts)은 여러 인공 신경망(전문가)의 예측들을 게이터(gater) 하나로 모아 한 과제를 수행하는 전략입니다.](http://www.bcs.rochester.edu/people/robbie/jacobslab/cheat_sheet/mixture_experts.pdf)
인자로 `dim`이 제공되면, `MixtureTable`은 보간될(도는 섞일) `전문가들` `텐서`의 차원을 특정합니다.
그렇지 않으면, `전문가들`은 `텐서`들로 구성된 `테이블` 하나를 입력으로 받해야 합니다. 
이 모듈은 1차원 이상의 `전문가들`을 위해, 그리고 1차원 또는 2차원 `게이터` 하나를 위해 동작합니다.
다시 말해, 단일 예제들 또는 미니배치들을 위해 동작합니다.

단일 예제 하나를 가진 `input = {G, E}` 하나를 고려하면,
게이터 `텐서` `G`를 가진 전문가들의 혼합물 `텐서` `E`는 다음과 같은 형식을 가집니다:
```lua
output = G[1]*E[1] + G[2]*E[2] + ... + G[n]*E[n]
```
여기서 `dim = 1`, `n = E:size(dim) = G:size(dim)`, 그리고 `G:dim() == 1`.
`E:dim() >= 2`임에 유의하십시오, `output:dim() = E:dim() - 1`.

예 1:
이 모듈을 사용하여, 2층 게이터를 가진 임의의 `n`개 2층 전문가들의 혼합물은 
다음과 같이 만들어질 수 있습니다:
```lua
experts = nn.ConcatTable()
for i = 1, n do
   local expert = nn.Sequential()
   expert:add(nn.Linear(3, 4))
   expert:add(nn.Tanh())
   expert:add(nn.Linear(4, 5))
   expert:add(nn.Tanh())
   experts:add(expert)
end

gater = nn.Sequential()
gater:add(nn.Linear(3, 7))
gater:add(nn.Tanh())
gater:add(nn.Linear(7, n))
gater:add(nn.SoftMax())

trunk = nn.ConcatTable()
trunk:add(gater)
trunk:add(experts)

moe = nn.Sequential()
moe:add(trunk)
moe:add(nn.MixtureTable())
```
두 예제로 구성된 배치 하나를 포워딩하면 다음과 같이 출력합니다:
```lua
> =moe:forward(torch.randn(2, 3))
-0.2152  0.3141  0.3280 -0.3772  0.2284
 0.2568  0.3511  0.0973 -0.0912 -0.0599
[torch.DoubleTensor of dimension 2x5]
```

예 2:
다음에서, `MixtureTable`은 `experts`로 `size = {1, 4, 2, 5, n}`인 `텐서` 하나를 기대합니다:
```lua
experts = nn.Concat(5)
for i = 1, n do
   local expert = nn.Sequential()
   expert:add(nn.Linear(3, 4))
   expert:add(nn.Tanh())
   expert:add(nn.Linear(4, 2*5))
   expert:add(nn.Tanh())
   expert:add(nn.Reshape(4, 2, 5, 1))
   experts:add(expert)
end

gater = nn.Sequential()
gater:add(nn.Linear(3, 7))
gater:add(nn.Tanh())
gater:add(nn.Linear(7, n))
gater:add(nn.SoftMax())

trunk = nn.ConcatTable()
trunk:add(gater)
trunk:add(experts)

moe = nn.Sequential()
moe:add(trunk)
moe:add(nn.MixtureTable(5))
```
두 예제로 구성된 배치 하나를 포워딩하면 다음과 같이 출력합니다:
```lua
> =moe:forward(torch.randn(2, 3)):size()
 2
 4
 2
 5
[torch.LongStorage of size 4]

```

<a name="nn.SelectTable"></a>
## SelectTable ##

`module` = `SelectTable(index)`

`테이블` 하나를 입력으로 받아서 인덱스 `index`(양수 또는 음수)에 있는 요소를 출력하는 
모듈 하나를 만듭니다.
이것은 `테이블` 하나 또는 `텐서` 하나일 수 있습니다.

비-`index` 요소들의 기울기들은 같은 차원을 가진 `0`으로 초기화된 `텐서`들입니다.
이것은 함수로서 캡슐화된 `텐서`의 깊이에 상관없이 사실입니다.
그렇게 하기 위해 내부적으로 사용된
는 반복적입니다.

The gradients of the non-`index` elements are zeroed `Tensor`s of the same size. 
This is true regardless of the depth of the encapsulated `Tensor` as the function 
used internally to do so is recursive.

예 1:
```lua
> input = {torch.randn(2, 3), torch.randn(2, 1)}
> =nn.SelectTable(1):forward(input)
-0.3060  0.1398  0.2707
 0.0576  1.5455  0.0610
[torch.DoubleTensor of dimension 2x3]

> =nn.SelectTable(-1):forward(input)
 2.3080
-0.2955
[torch.DoubleTensor of dimension 2x1]

> =table.unpack(nn.SelectTable(1):backward(input, torch.randn(2, 3)))
-0.4891 -0.3495 -0.3182
-2.0999  0.7381 -0.5312
[torch.DoubleTensor of dimension 2x3]

0
0
[torch.DoubleTensor of dimension 2x1]

```

예 2:
```lua
> input = {torch.randn(2, 3), {torch.randn(2, 1), {torch.randn(2, 2)}}}

> =nn.SelectTable(2):forward(input)
{
  1 : DoubleTensor - size: 2x1
  2 :
    {
      1 : DoubleTensor - size: 2x2
    }
}

> =table.unpack(nn.SelectTable(2):backward(input, {torch.randn(2, 1), {torch.randn(2, 2)}}))
0 0 0
0 0 0
[torch.DoubleTensor of dimension 2x3]

{
  1 : DoubleTensor - size: 2x1
  2 :
    {
      1 : DoubleTensor - size: 2x2
    }
}

> gradInput = nn.SelectTable(1):backward(input, torch.randn(2, 3))

> =gradInput
{
  1 : DoubleTensor - size: 2x3
  2 :
    {
      1 : DoubleTensor - size: 2x1
      2 :
        {
          1 : DoubleTensor - size: 2x2
        }
    }
}

> =gradInput[1]
-0.3400 -0.0404  1.1885
 1.2865  0.4107  0.6506
[torch.DoubleTensor of dimension 2x3]

> gradInput[2][1]
0
0
[torch.DoubleTensor of dimension 2x1]

> gradInput[2][2][1]
0 0
0 0
[torch.DoubleTensor of dimension 2x2]

```

<a name="nn.NarrowTable"></a>
## NarrowTable ##

`module` = `NarrowTable(offset [, length])`

Creates a module that takes a `table` as input and outputs the subtable 
starting at index `offset` having `length` elements (defaults to 1 element).
The elements can be either a `table` or a [`Tensor`](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor).

The gradients of the elements not included in the subtable are zeroed `Tensor`s of the same size. 
This is true regardless of the dept of the encapsulated `Tensor` as the function used internally to do so is recursive.

Example:
```lua
> input = {torch.randn(2, 3), torch.randn(2, 1), torch.randn(1, 2)}
> =nn.NarrowTable(2,2):forward(input)
{
  1 : DoubleTensor - size: 2x1
  2 : DoubleTensor - size: 1x2
}

> =nn.NarrowTable(1):forward(input)
{
  1 : DoubleTensor - size: 2x3
}

> =table.unpack(nn.NarrowTable(1,2):backward(input, {torch.randn(2, 3), torch.randn(2, 1)}))
 1.9528 -0.1381  0.2023
 0.2297 -1.5169 -1.1871
[torch.DoubleTensor of size 2x3]

-1.2023
-0.4165
[torch.DoubleTensor of size 2x1]

 0  0
[torch.DoubleTensor of size 1x2]

```

<a name="nn.FlattenTable"></a>
## FlattenTable ##

`module` = `FlattenTable()`

Creates a module that takes an arbitrarily deep `table` of `Tensor`s (potentially nested) as input and outputs a `table` of `Tensor`s, where the output `Tensor` in index `i` is the `Tensor` with post-order DFS index `i` in the input `table`.

This module is particularly useful in combination with nn.Identity() to create networks that can append to their input `table`.

Example:
```lua
x = {torch.rand(1), {torch.rand(2), {torch.rand(3)}}, torch.rand(4)}
print(x)
print(nn.FlattenTable():forward(x))
```
gives the output:
```lua
{
  1 : DoubleTensor - size: 1
  2 :
    {
      1 : DoubleTensor - size: 2
      2 :
        {
          1 : DoubleTensor - size: 3
        }
    }
  3 : DoubleTensor - size: 4
}
{
  1 : DoubleTensor - size: 1
  2 : DoubleTensor - size: 2
  3 : DoubleTensor - size: 3
  4 : DoubleTensor - size: 4
}
```

<a name="nn.PairwiseDistance"></a>
## PairwiseDistance ##

`module` = `PairwiseDistance(p)` creates a module that takes a `table` of two vectors as input and outputs the distance between them using the `p`-norm.

Example:
```lua
mlp_l1 = nn.PairwiseDistance(1)
mlp_l2 = nn.PairwiseDistance(2)
x = torch.Tensor({1, 2, 3})
y = torch.Tensor({4, 5, 6})
print(mlp_l1:forward({x, y}))
print(mlp_l2:forward({x, y}))
```
gives the output:
```lua
 9
[torch.Tensor of dimension 1]

 5.1962
[torch.Tensor of dimension 1]
```

A more complicated example:
```lua
-- imagine we have one network we are interested in, it is called "p1_mlp"
p1_mlp= nn.Sequential(); p1_mlp:add(nn.Linear(5, 2))

-- But we want to push examples towards or away from each other
-- so we make another copy of it called p2_mlp
-- this *shares* the same weights via the set command, but has its own set of temporary gradient storage
-- that's why we create it again (so that the gradients of the pair don't wipe each other)
p2_mlp= nn.Sequential(); p2_mlp:add(nn.Linear(5, 2))
p2_mlp:get(1).weight:set(p1_mlp:get(1).weight)
p2_mlp:get(1).bias:set(p1_mlp:get(1).bias)

-- we make a parallel table that takes a pair of examples as input. they both go through the same (cloned) mlp
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

-- now we define our top level network that takes this parallel table and computes the pairwise distance betweem
-- the pair of outputs
mlp= nn.Sequential()
mlp:add(prl)
mlp:add(nn.PairwiseDistance(1))

-- and a criterion for pushing together or pulling apart pairs
crit = nn.HingeEmbeddingCriterion(1)

-- lets make two example vectors
x = torch.rand(5)
y = torch.rand(5)


-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
local pred = mlp:forward(x)
local err = criterion:forward(pred, y)
local gradCriterion = criterion:backward(pred, y)
mlp:zeroGradParameters()
mlp:backward(x, gradCriterion)
mlp:updateParameters(learningRate)
end

-- push the pair x and y together, notice how then the distance between them given
-- by  print(mlp:forward({x, y})[1]) gets smaller
for i = 1, 10 do
gradUpdate(mlp, {x, y}, 1, crit, 0.01)
print(mlp:forward({x, y})[1])
end


-- pull apart the pair x and y, notice how then the distance between them given
-- by  print(mlp:forward({x, y})[1]) gets larger

for i = 1, 10 do
gradUpdate(mlp, {x, y}, -1, crit, 0.01)
print(mlp:forward({x, y})[1])
end

```

<a name="nn.DotProduct"></a>
## DotProduct ##

`module` = `DotProduct()` creates a module that takes a `table` of two vectors as input and outputs the dot product between them.

Example:
```lua
mlp = nn.DotProduct()
x = torch.Tensor({1, 2, 3})
y = torch.Tensor({4, 5, 6})
print(mlp:forward({x, y}))
```
gives the output:
```lua
 32
[torch.Tensor of dimension 1]
```


A more complicated example:
```lua

-- Train a ranking function so that mlp:forward({x, y}, {x, z}) returns a number
-- which indicates whether x is better matched with y or z (larger score = better match), or vice versa.

mlp1 = nn.Linear(5, 10)
mlp2 = mlp1:clone('weight', 'bias')

prl = nn.ParallelTable();
prl:add(mlp1); prl:add(mlp2)

mlp1 = nn.Sequential()
mlp1:add(prl)
mlp1:add(nn.DotProduct())

mlp2 = mlp1:clone('weight', 'bias')

mlp = nn.Sequential()
prla = nn.ParallelTable()
prla:add(mlp1)
prla:add(mlp2)
mlp:add(prla)

x = torch.rand(5);
y = torch.rand(5)
z = torch.rand(5)


print(mlp1:forward{x, x})
print(mlp1:forward{x, y})
print(mlp1:forward{y, y})


crit = nn.MarginRankingCriterion(1);

-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

inp = {{x, y}, {x, z}}

math.randomseed(1)

-- make the pair x and y have a larger dot product than x and z

for i = 1, 100 do
   gradUpdate(mlp, inp, 1, crit, 0.05)
   o1 = mlp1:forward{x, y}[1];
   o2 = mlp2:forward{x, z}[1];
   o = crit:forward(mlp:forward{{x, y}, {x, z}}, 1)
   print(o1, o2, o)
end

print "________________**"

-- make the pair x and z have a larger dot product than x and y

for i = 1, 100 do
   gradUpdate(mlp, inp, -1, crit, 0.05)
   o1 = mlp1:forward{x, y}[1];
   o2 = mlp2:forward{x, z}[1];
   o = crit:forward(mlp:forward{{x, y}, {x, z}}, -1)
   print(o1, o2, o)
end
```


<a name="nn.CosineDistance"></a>
## CosineDistance ##

`module` = `CosineDistance()` creates a module that takes a `table` of two vectors (or matrices if in batch mode) as input and outputs the cosine distance between them.

Examples:
```lua
mlp = nn.CosineDistance()
x = torch.Tensor({1, 2, 3})
y = torch.Tensor({4, 5, 6})
print(mlp:forward({x, y}))
```
gives the output:
```lua
 0.9746
[torch.Tensor of dimension 1]
```
`CosineDistance` also accepts batches:
```lua
mlp = nn.CosineDistance()
x = torch.Tensor({{1,2,3},{1,2,-3}})
y = torch.Tensor({{4,5,6},{-4,5,6}})
print(mlp:forward({x,y}))
```
gives the output:
```lua
 0.9746
-0.3655
[torch.DoubleTensor of size 2]
```

A more complicated example:
```lua

-- imagine we have one network we are interested in, it is called "p1_mlp"
p1_mlp= nn.Sequential(); p1_mlp:add(nn.Linear(5, 2))

-- But we want to push examples towards or away from each other
-- so we make another copy of it called p2_mlp
-- this *shares* the same weights via the set command, but has its own set of temporary gradient storage
-- that's why we create it again (so that the gradients of the pair don't wipe each other)
p2_mlp= p1_mlp:clone('weight', 'bias')

-- we make a parallel table that takes a pair of examples as input. they both go through the same (cloned) mlp
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

-- now we define our top level network that takes this parallel table and computes the cosine distance betweem
-- the pair of outputs
mlp= nn.Sequential()
mlp:add(prl)
mlp:add(nn.CosineDistance())


-- lets make two example vectors
x = torch.rand(5)
y = torch.rand(5)

-- Grad update function..
function gradUpdate(mlp, x, y, learningRate)
    local pred = mlp:forward(x)
    if pred[1]*y < 1 then
        gradCriterion = torch.Tensor({-y})
        mlp:zeroGradParameters()
        mlp:backward(x, gradCriterion)
        mlp:updateParameters(learningRate)
    end
end

-- push the pair x and y together, the distance should get larger..
for i = 1, 1000 do
 gradUpdate(mlp, {x, y}, 1, 0.1)
 if ((i%100)==0) then print(mlp:forward({x, y})[1]);end
end


-- pull apart the pair x and y, the distance should get smaller..

for i = 1, 1000 do
 gradUpdate(mlp, {x, y}, -1, 0.1)
 if ((i%100)==0) then print(mlp:forward({x, y})[1]);end
end
```



<a name="nn.CriterionTable"></a>
## CriterionTable ##

`module` = `CriterionTable(criterion)`

Creates a module that wraps a Criterion module so that it can accept a `table` of inputs. Typically the `table` would contain two elements: the input and output `x` and `y` that the Criterion compares.

Example:
```lua
mlp = nn.CriterionTable(nn.MSECriterion())
x = torch.randn(5)
y = torch.randn(5)
print(mlp:forward{x, x})
print(mlp:forward{x, y})
```
gives the output:
```lua
0
1.9028918413199
```

Here is a more complex example of embedding the criterion into a network:
```lua

function table.print(t)
 for i, k in pairs(t) do print(i, k); end
end

mlp = nn.Sequential();                          -- Create an mlp that takes input
  main_mlp = nn.Sequential();		      -- and output using ParallelTable
  main_mlp:add(nn.Linear(5, 4))
  main_mlp:add(nn.Linear(4, 3))
 cmlp = nn.ParallelTable();
 cmlp:add(main_mlp)
 cmlp:add(nn.Identity())
mlp:add(cmlp)
mlp:add(nn.CriterionTable(nn.MSECriterion())) -- Apply the Criterion

for i = 1, 20 do                                 -- Train for a few iterations
 x = torch.ones(5);
 y = torch.Tensor(3); y:copy(x:narrow(1, 1, 3))
 err = mlp:forward{x, y}                         -- Pass in both input and output
 print(err)

 mlp:zeroGradParameters();
 mlp:backward({x, y} );
 mlp:updateParameters(0.05);
end
```

<a name="nn.CAddTable"></a>
## CAddTable ##

Takes a `table` of `Tensor`s and outputs summation of all `Tensor`s.

```lua
ii = {torch.ones(5), torch.ones(5)*2, torch.ones(5)*3}
=ii[1]
 1
 1
 1
 1
 1
[torch.DoubleTensor of dimension 5]

return ii[2]
 2
 2
 2
 2
 2
[torch.DoubleTensor of dimension 5]

return ii[3]
 3
 3
 3
 3
 3
[torch.DoubleTensor of dimension 5]

m = nn.CAddTable()
=m:forward(ii)
 6
 6
 6
 6
 6
[torch.DoubleTensor of dimension 5]
```


<a name="nn.CSubTable"></a>
## CSubTable ##

Takes a `table` with two `Tensor` and returns the component-wise
subtraction between them.

```lua
m = nn.CSubTable()
=m:forward({torch.ones(5)*2.2, torch.ones(5)})
 1.2000
 1.2000
 1.2000
 1.2000
 1.2000
[torch.DoubleTensor of dimension 5]
```

<a name="nn.CMulTable"></a>
## CMulTable ##

Takes a `table` of `Tensor`s and outputs the multiplication of all of them.

```lua
ii = {torch.ones(5)*2, torch.ones(5)*3, torch.ones(5)*4}
m = nn.CMulTable()
=m:forward(ii)
 24
 24
 24
 24
 24
[torch.DoubleTensor of dimension 5]

```

<a name="nn.CDivTable"></a>
## CDivTable ##

Takes a `table` with two `Tensor` and returns the component-wise
division between them.

```lua
m = nn.CDivTable()
=m:forward({torch.ones(5)*2.2, torch.ones(5)*4.4})
 0.5000
 0.5000
 0.5000
 0.5000
 0.5000
[torch.DoubleTensor of dimension 5]
```

