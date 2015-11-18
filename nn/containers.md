<a name="nn.Containers"></a>
# 컨테이너 #
컨테이너 클래스들을 사용하면, 복잡한 신경망도 쉽게 만들 수 있습니다.

  * [컨테이너](#nn.Container) : 컨테이너들에 의해 상속되는 추상 클래스 ;
    * [Sequential](#nn.Sequential) : 층들을 하나의 피드포워드 완전 연결 방식으로 연결합니다 ;
    * [Parallel](#nn.Parallel) : 그것의 `i` 번째 자식 모듈을 입력 텐서의 `i` 번째 단면에 적용합니다 ;
    * [Concat](#nn.Concat) : 한 층에서 차원 `dim`을 따라 여러 모듈들을 이어붙입니다 ;
      * [DepthConcat](#nn.DepthConcat) : Concat과 같습니다. 그러나 차원 `dim`을 제외한 차원들의 `sizes`가 매치되지 않을 때, zero-padding을 추가합니다 ;
 
또한, [텐서](https://github.com/torch/torch7/blob/master/doc/tensor.md)들의 테이블 조작을 위해 [테이블 컨테이너](#nn.TableContainers)도 보십시오.

<a name="nn.Container"></a>
## 컨테이너 ##

이것은 모든 컨테이너들에서 정의되는 메소스들을 선언하는 추상 [모듈](module.md#nn.Module) 클래스입니다. 
이것은 (그런 호출들이 포함된 모듈들로 전파되는) 많은 `Module` 메소드들을 다시 구현합니다.
예를 들어, [zeroGradParameters](module.md#nn.Module.zeroGradParameters) 호출은 
모든 포함된 모듈들로 전파될 것입니다.


<a name="nn.Container.add"></a>
### add(module) ###
주어진 `module`을 그 컨테이너에 추가합니다. 그 순서 또한 중요합니다.

<a name="nn.Container.get"></a>
### get(index) ###
인덱스 `index`에서의 포함된 모듈들을 리턴합니다.

<a name="nn.Container.size"></a>
### size() ###
그 포함된 모듈들의 개수를 리턴합니다.

<a name="nn.Sequential"></a>
## Sequential ##

`Sequential`은 층들을 하나의 피드포워드 완전 연결 방식으로 연결하는 수단을 제공합니다.

예를 들어, 숨겨진 층이 하나인 다층 퍼셉트론을 만드는 것은 다음과 같이 쉽습니다.
```lua
mlp = nn.Sequential()
mlp:add( nn.Linear(10, 25) ) -- 입력 10개, 숨겨진 유닛 25개
mlp:add( nn.Tanh() ) -- 쌍곡탄젠트(hyperbolic tangent) 전달 함수
mlp:add( nn.Linear(25, 1) ) -- 출력 1개

mlp:forward(torch.randn(10))
```
그 코드의 출력은 다음과 같습니다.
```lua
-0.1815
[torch.Tensor of dimension 1]
```

<a name="nn.Sequential.remove"></a>
### remove([index]) ###

주어진 `index`에 있는 모듈을 없앱니다. 만약 `index`가 특정되지 않으면, 마지막 층을 없앱니다.

```lua
model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 20))
model:add(nn.Linear(20, 30))
model:remove(2)
> model
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.Linear(10 -> 20)
  (2): nn.Linear(20 -> 30)
}
```


<a name="nn.Sequential.insert"></a>
### insert(module, [index]) ###

주어진 `module`을 주어진 `index`에 끼워넣습니다. 만약 `index`가 특정되지 않으면, 시퀀스의 길이를 늘려 가장 마지막 인덱스에 그 `module`을 추가합니다.

```lua
model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 30))
model:insert(nn.Linear(20, 20), 2)
> model
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.Linear(10 -> 20)
  (2): nn.Linear(20 -> 20)      -- 끼워넣어진 층
  (3): nn.Linear(20 -> 30)
}
```



<a name="nn.Parallel"></a>
## Parallel ##

`module` = `Parallel(inputDimension,outputDimension)`

`Parallel` 컨테이너의 입력으로 들어가는 입력 텐서가 하나 있습니다.
그 입력 텐서는 `inputDimension` 차원을 따라  [select](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-selectdim-index)를 사용하여 
단면들로 나눠져 `Parallel` 컨테이너의 각 모듈로 입력됩니다. 예를 들어, `i` 번째 단면은 `Parallel` 컨테이너의 `i` 번째 자식 모듈로 입력됩니다. `Parallel` 컨테이너는 거기에 포함된 모듈들의 결과를 차원 `outputDimension`을 따라 이어붙입니다.

예:
```lua
 mlp=nn.Parallel(2,1)      -- 입력의 차원 2에 대해 반복함
 mlp:add(nn.Linear(10,3))  -- 첫 번째 슬라이스 적용
 mlp:add(nn.Linear(10,2))  -- 두 번째 슬라이스 적용
 mlp:forward(torch.randn(10,2))
```
는 다음을 출력합니다:
```lua
-0.5300
-1.1015
 0.7764
 0.2819
-0.6026
[torch.Tensor of dimension 5]
```

더 복잡한 예제:
```lua
mlp=nn.Sequential();
c=nn.Parallel(1,2)
for i=1,10 do
 local t=nn.Sequential()
 t:add(nn.Linear(3,2))
 t:add(nn.Reshape(2,1))
 c:add(t)
end
mlp:add(c)

pred=mlp:forward(torch.randn(10,3))
print(pred)

for i=1,10000 do     -- 적은 수의 반복으로 훈련
 x=torch.randn(10,3);
 y=torch.ones(2,10);
 pred=mlp:forward(x)

 criterion= nn.MSECriterion()
 local err=criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.01);
 print(err)
end
```


<a name="nn.Concat"></a>
## Concat ##

```lua
module = nn.Concat(dim)
```
`Concat`은 한 층에 있는 `parallel` 모듈들의 출력을 차원 `dim`을 따라 이어붙입니다.
`Concat`은 같은 입력을 받고, 그 출력은 이어붙여집니다.

```lua
mlp=nn.Concat(1);
mlp:add(nn.Linear(5,3))
mlp:add(nn.Linear(5,7))
print(mlp:forward(torch.randn(5)))
```
는 다음 결과를 출력합니다:
```lua
 0.7486
 0.1349
 0.7924
-0.0371
-0.4794
 0.3044
-0.0835
-0.7928
 0.7856
-0.1815
[torch.Tensor of dimension 10]
```

<a name="nn.DepthConcat"></a>
## DepthConcat ##

```lua
module = nn.DepthConcat(dim)
```
`DepthConcat`은 한 층에 있는 "parallel" 모듈들의 출력을 차원 `dim`을 따라 이어붙입니다.
`DepthConcat`은 같은 입력을 받습니다, 그리고 그 출력은 이어붙여집니다. 
`dim`을 제외한, 다른 `sizes(차원수)`를 가진 차원들을 위해, 
출력의 텐서의 중심에 더 작은 텐서들이 복사됩니다.
이 연산은 경계들을 효과적으로 `0`들로 채웁니다.

그 모듈은 [Convolutions](convolution.md)의 출력을 깊이 차원(다시 말해, `nOutputFrame`)을 따라 
이어붙이는 데 특히 유용합니다. 
이것은 [Going deeper with convolutions](http://arxiv.org/pdf/1409.4842v1.pdf) 논문의 
*DepthConcat* 층을 구현하는 데 사용됩니다. 
보통의 [Concat](#nn.Concat) 모듈은 사용될 수 없습니다. 
왜냐하면 이어붙여져야 하는 출력 텐서들의 공간적 차원들(높이와 넓이)이 
다른 값들을 가질 수도 있기 때문입니다. 이 문제를 다루기 위해, 출력은 
가장 큰 공간적 차원들을 사용합니다, 그리고 더 작은 텐서들 주변을 `0`으로 채웁니다.

```lua
inputSize = 3
outputSize = 2
input = torch.randn(inputSize,7,7)
mlp=nn.DepthConcat(1);
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 1, 1))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 3, 3))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 4, 4))
print(mlp:forward(input))
```
는 다음을 출력합니다:
```lua
(1,.,.) = 
 -0.2874  0.6255  1.1122  0.4768  0.9863 -0.2201 -0.1516
  0.2779  0.9295  1.1944  0.4457  1.1470  0.9693  0.1654
 -0.5769 -0.4730  0.3283  0.6729  1.3574 -0.6610  0.0265
  0.3767  1.0300  1.6927  0.4422  0.5837  1.5277  1.1686
  0.8843 -0.7698  0.0539 -0.3547  0.6904 -0.6842  0.2653
  0.4147  0.5062  0.6251  0.4374  0.3252  0.3478  0.0046
  0.7845 -0.0902  0.3499  0.0342  1.0706 -0.0605  0.5525

(2,.,.) = 
 -0.7351 -0.9327 -0.3092 -1.3395 -0.4596 -0.6377 -0.5097
 -0.2406 -0.2617 -0.3400 -0.4339 -0.3648  0.1539 -0.2961
 -0.7124 -1.2228 -0.2632  0.1690  0.4836 -0.9469 -0.7003
 -0.0221  0.1067  0.6975 -0.4221 -0.3121  0.4822  0.6617
  0.2043 -0.9928 -0.9500 -1.6107  0.1409 -1.3548 -0.5212
 -0.3086 -0.0298 -0.2031  0.1026 -0.5785 -0.3275 -0.1630
  0.0596 -0.6097  0.1443 -0.8603 -0.2774 -0.4506 -0.5367

(3,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000 -0.7326  0.3544  0.1821  0.4796  1.0164  0.0000
  0.0000 -0.9195 -0.0567 -0.1947  0.0169  0.1924  0.0000
  0.0000  0.2596  0.6766  0.0939  0.5677  0.6359  0.0000
  0.0000 -0.2981 -1.2165 -0.0224 -1.1001  0.0008  0.0000
  0.0000 -0.1911  0.2912  0.5092  0.2955  0.7171  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(4,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000 -0.8263  0.3646  0.6750  0.2062  0.2785  0.0000
  0.0000 -0.7572  0.0432 -0.0821  0.4871  1.9506  0.0000
  0.0000 -0.4609  0.4362  0.5091  0.8901 -0.6954  0.0000
  0.0000  0.6049 -0.1501 -0.4602 -0.6514  0.5439  0.0000
  0.0000  0.2570  0.4694 -0.1262  0.5602  0.0821  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(5,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.3158  0.4389 -0.0485 -0.2179  0.0000  0.0000
  0.0000  0.1966  0.6185 -0.9563 -0.3365  0.0000  0.0000
  0.0000 -0.2892 -0.9266 -0.0172 -0.3122  0.0000  0.0000
  0.0000 -0.6269  0.5349 -0.2520 -0.2187  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(6,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  1.1148  0.2324 -0.1093  0.5024  0.0000  0.0000
  0.0000 -0.2624 -0.5863  0.3444  0.3506  0.0000  0.0000
  0.0000  0.1486  0.8413  0.6229 -0.0130  0.0000  0.0000
  0.0000  0.8446  0.3801 -0.2611  0.8140  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
[torch.DoubleTensor of dimension 6x7x7]
```

어떻게 6개 필터 맵 중 마지막 2개가 그 왼쪽과 위에 제로패딩(zero-padding) 1열(또는 행)을 가지는지,
그리고 그 오른쪽과 아래에 제로패딩 2열(또는 행)을 가지는지에 유념하십시오.
요소 모듈 출력 텐서들에서 `dim` 차원을 뺀 나머지 차원수들이 모두 홀수 또는 짝수가 아닐 때, 
이 제로패딩은 피할 수 없습니다.
매핑이 정렬되도록 유지하기 위해, 우리는 이것들이 모두 홀수 (또는 짝수)임을 보증해야할 필요가 있습니다.

<a name="nn.TableContainers"></a>
## 테이블 컨테이너 ##
위 컨테이너들이 입력 [텐서](https://github.com/torch/torch7/blob/master/doc/tensor.md)들을 조작하기 위해 사용되는 반면, 테이블 컨테이너들은 테이블들을 조작하기 위해 사용됩니다.
 * [ConcatTable](table.md#nn.ConcatTable)
 * [ParallelTable](table.md#nn.ParallelTable)

이 컨테이너들은 테이블들을 조작하는 모든 다른 모듈들과 함께 [여기](table.md)에서 설명됩니다.
