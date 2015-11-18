<a name="nn.Criterions"></a>
# 기준들 (Criterions) #

[`기준들`]은 신경망을 훈련시키는 데 도움이 됩니다. 
주어진 입력과 타겟으로, 기준들은 주어진 손실 함수에 따라 기울기를 계산합니다.

  * 분류 기준들:
    * [`BCECriterion`](#nn.BCECriterion): [`Sigmoid`](transfer.md#nn.Sigmoid)를 위한 바이너리 크로스-엔트로피  ([`ClassNLLCriterion`](#nn.ClassNLLCriterion)의 두 부류 버전);
    * [`ClassNLLCriterion`](#nn.ClassNLLCriterion): [`LogSoftMax`](transfer.md#nn.LogSoftMax)를 위한 네거티브 로그-우도 (다-부류);
    * [`CrossEntropyCriterion`](#nn.CrossEntropyCriterion): [`LogSoftMax`](transfer.md#nn.LogSoftMax)와 [`ClassNLLCriterion`](#nn.ClassNLLCriterion)를 결합합니다;
    * [`MarginCriterion`](#nn.MarginCriterion): 두 부류 마진(margin) 기반 손실;
    * [`MultiMarginCriterion`](#nn.MultiMarginCriterion): 다-부류 마진 기반 손실;
    * [`MultiLabelMarginCriterion`](#nn.MultiLabelMarginCriterion): 다-부류 다-분류 마진 기반 손실;
  * 회귀 기준들:
    * [`AbsCriterion`](#nn.AbsCriterion): 입력 사이 요소별 차이의 절댓값의 평균을 측정합니다;
    * [`SmoothL1Criterion`](#nn.SmoothL1Criterion): AbsCriterion의 부드러운(smooth) 버전;
    * [`MSECriterion`](#nn.MSECriterion): 평균 제곱 오차 (대표적인);
    * [`DistKLDivCriterion`](#nn.DistKLDivCriterion): 쿨백-라이블러 발산 (연속 확률 분포를 피팅(fitting)하기 위한);
  * 임베딩 기준들 (두 입력이 비슷한지 안 비슷한지 특정하는):
    * [`HingeEmbeddingCriterion`](#nn.HingeEmbeddingCriterion): 입력으로 거리를 받음;
    * [`L1HingeEmbeddingCriterion`](#nn.L1HingeEmbeddingCriterion): 두 입력 사이 L1 거리;
    * [`CosineEmbeddingCriterion`](#nn.CosineEmbeddingCriterion): 두 입력 사이 코사인 거리;
  * 그 밖의 기준들:
    * [`MultiCriterion`](#nn.MultiCriterion) : 같은 입력과 타겟에 적용되는 각각 다른 기준들의 가중된 합.
    * [`ParallelCriterion`](#nn.ParallelCriterion) : 다른 입력과 타겟에 적용되는 각각 다른 기준들의 가중된 합.
    * [`MarginRankingCriterion`](#nn.MarginRankingCriterion): 두 입력을 랭크(rank)합니다;

<a name="nn.Criterion"></a>
## 기준 (Criterion) ##

이것은 모든 기준들에서 정의되는 메소드들을 선언하는 추상 클래스입니다.
이 클래스는 [직렬화](https://github.com/torch/torch7/blob/master/doc/file.md#serialization-methods)할 수 있습니다.

<a name="nn.Criterion.forward"></a>
### [output] forward(input, target) ###

주어진 `input`과 `target`으로, 기준과 관련된 손실 함수를 계산하고 그 결과를 리턴합니다.
일반적으로 `input`과 `target`은 [`텐서`](https://github.com/torch/torch7/blob/master/doc/tensor.md)입니다.
그러나 몇몇 특정 기준들은 다른 타입의 객체를 요구할 수도 있습니다.

리턴되는 `output`은 일반적으로 스칼라 하나입니다.

상태 변수 [`self.output`](#nn.Criterion.output)은 `forward()` 호출 뒤에 꼭 갱신되어야 합니다.


<a name="nn.Criterion.backward"></a>
### [gradInput] backward(input, target) ###

주어진 `input`과 `target`으로, 기준에 관련된 손실 함수의 기울기들을 계산하고 그 결과를 리턴합니다.
일반적으로 `input`, `target`, 그리고 `gradInput`은 [`텐서`](https://github.com/torch/torch7/blob/master/doc/tensor.md)입니다.
그러나 몇몇 특정 기준들은 다른 타입의 객체를 요구할 수도 있습니다.

상태 변수 [`self.gradInput`](#nn.Criterion.gradInput)은 `backward()` 호출 뒤에 꼭 갱신되어야 합니다.


<a name="nn.Criterion.output"></a>
### 상태 변수: output ###

마지막 [`forward(input, target)`](#nn.Criterion.forward) 호출의 결과를 담은 상태 변수.


<a name="nn.Criterion.gradInput"></a>
### 상태 변수: gradInput ###

마지막 [`backward(input, target)`](#nn.Criterion.backward) 호출의 결과를 담은 상태 변수.


<a name="nn.AbsCriterion"></a>
## AbsCriterion ##

```lua
criterion = nn.AbsCriterion()
```
입력 `x`와 타겟 `y` 사이 요소별 차이의 절댓값의 평균을 측정하는 기준을 만듭니다.

```lua
loss(x, y)  = 1/n \sum |x_i - y_i|
```

만약 `x`와 `y`가 전체 요소가 `n`개로 구성된 `d`차원 텐서이면, 
합 연산이 여전이 모든 요소에 대해 동작합니다, 그리고 `n`으로 나눕니다.

만약 내부 변수 `sizeAverage`가 `false`로 설정되면, `n`으로 나누지 않도록 할 수 있습니다.

```lua
criterion = nn.AbsCriterion()
criterion.sizeAverage = false
```


<a name="nn.ClassNLLCriterion"></a>
## ClassNLLCriterion ##

```lua
criterion = nn.ClassNLLCriterion([weights])
```
네거티브 로그 우도 기준. 이것은 `n` 부류 분류 문제로 훈련시키는 데 유용합니다.
만약 제공되면, 선택적 인자 `weights`는 각 클래스에 가중치를 할당하는 1차원 `텐서`여야 합니다.
이것은 당신이 균형 잡히지 않은(unbalanced) 훈련 집합을 가질 때, 특히 유용합니다.

`forward()`로 주어지는 `input`은 각 클래스의 _log-probabilities_를 포함하고 있을 것으로 기대됩니다:
`input`은 반드시 크기 `n`인 1차원 `텐서`여야 합니다.
신경망에서 로그 확률들은 당신의 신경망 마지막 층에 [`LogSoftMax`](#nn.LogSoftMax) 층을 더함으로써 
쉽게 얻어질 수 있습니다.
만약 당신이 네트워크에 추가적인 층을 덧붙이고 싶지 않으면,
대신 [`CrossEntropyCriterion`](#nn.CrossEntropyCriterion)가 사용될 수도 있습니다.
[`forward(input, target`)](#nn.CriterionForward)와 [`backward(input, target)`](#nn.CriterionBackward)를
호출할 때,
이 기준은 `target`으로 한 클래스 인덱스(1부터 클래스 개수)를 기대합니다.

손실은 다음 같이 설명될 수 있습니다:

```lua
loss(x, class) = -x[class]
```

또는 `weights` 인자가 특졍되는 경우:

```lua
loss(x, class) = -weights[class] * x[class]
```

다음은 어떻게 기울기 단계를 만드는지 보여주는 짧은 코드입니다.
인자들은 다음과 같이 주어집니다. 입력 `x`, 희망 출력(타겟) `y` (정수 `1`부터 `n`, 이 경우 `n=2` 부류),
네트워크 `mlp`, 그리고 학습률 `learningRate`:

```lua
function gradUpdate(mlp, x, y, learningRate)
   local criterion = nn.ClassNLLCriterion()
   pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   mlp:zeroGradParameters()
   local t = criterion:backward(pred, y)
   mlp:backward(x, t)
   mlp:updateParameters(learningRate)
end
```


<a name="nn.CrossEntropyCriterion"></a>
## CrossEntropyCriterion ##

```lua
criterion = nn.CrossEntropyCriterion([weights])
```

이 기준은 한 단일 클래스에서 [`LogSoftMax`](#nn.LogSoftMax)와 
[`ClassNLLCriterion`](#nn.ClassNLLCriterion)을 합칩니다.

이것은 `n` 부류 분류 문제에 대해 신경망을 훈련시키는 데 유용합니다.
만약 제공되면, 선택적 인자 `weights`는 가중치를 각 부류에 할당하는 1차원 `텐서`여야 합니다.
이것은 당신이 균형 잡히지 않은(unbalanced) 훈련 집합을 가질 때, 특히 유용합니다.

`forward()`로 주어지는 `input`은 각 부류를 위한 점수들을 가지고 있을 것으로 기대됩니다:
`input`은 반드시 크기가 `n`인 1차원 `텐서`여야 합니다.
[`forward(input, target`)](#nn.CriterionForward)와 [`backward(input, target)`](#nn.CriterionBackward)를
호출할 때, 
이 기준은 `target`으로 한 클래스 인덱스(1부터 클래스 개수)를 기대합니다.

손실은 다음 같이 설명될 수 있습니다:

```lua
loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
               = -x[class] + log(\sum_j exp(x[j]))
```

또는 `weights` 인자가 특정되는 경우:

```lua
loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))
```


<a name="nn.DistKLDivCriterion"></a>
## DistKLDivCriterion ##

```lua
criterion = nn.DistKLDivCriterion()
```

[쿨백-라이블러 발산](https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0) 기준.
KL 발산은 연속 분포들을 위한 유용한 거리 척도입니다. 그리고 KL 발산은
(이산적으로 샘플된) 연속 출력 분포들의 공간에 대해 다이렉트 회귀(direct regression)를 수행할 때 종종 유용합니다.
`ClassNLLCriterion`과 마찬가지로 `forward()`로 주어지는 `input`은 _로그-확률들_을 담고 있을 것으로 기대됩니다.
그러나 `ClassNLLCriterion`와 달리, `input`은 1차원 또는 2차원 벡터로 제한되지 않습니다 (기준이 요소별로 적용되므로).

[`forward(input, target)`](#nn.CriterionForward)와 [`backward(input, target)`](#nn.CriterionBackward)를
호출할 때, 이 기준은 `input` `Tensor` 와 같은 차원의 `target` `Tensor` 하나를 기대합니다.

손실은 다음과 같이 설명될 수 있습니다:

```lua
loss(x, target) = \sum(target_i * (log(target_i) - x_i))
```


<a name="nn.BCECriterion"></a>
## BCECriterion

```lua
criterion = nn.BCECriterion([weights])
```

타겟과 출력 사이 2진 엔트로피를 측정하는 한 기준을 만듭니다:

```lua
loss(t, o) = - sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
```

또는 `weights` 인자가 특정되는 경우:

```lua
loss(t, o) = - sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
```

이를테면 오토인코더에서, 이것은 복원의 오차를 측정하는 데 사용됩니다.
타겟 `t[i]`는 [`nn.Sigmoid`](transfer.md#nn.Sigmoid) 층의 출력처럼 0과 1 사이 숫자여야 함에 주의하십시오.


<a name="nn.MarginCriterion"></a>
## MarginCriterion ##

```lua
criterion = nn.MarginCriterion([margin])
```

입력 `x`(차원 `1`인 `텐서`)와 출력 `y`(`1` 또는 `-1`들로 구성된 텐서) 사이의
두 부류 분류 힌지(hinge) 손실[마진 기반 손실]을 최적화하는 기준을 만듭니다.
만약 특정되지 않으면, `margin`의 기본값은 `1`입니다.

```lua
loss(x, y) = sum_i (max(0, margin - y[i]*x[i])) / x:nElement()
```

입력에 있는 요소 개수에 의한 정규화는 `self.sizeAverage`를 `false`로 설정함으로써
비활성화될 수 있습니다.


### 예

```lua
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
mlp:add(nn.Linear(5, 1))

x1 = torch.rand(5)
x1_target = torch.Tensor{1}
x2 = torch.rand(5)
x2_target = torch.Tensor{-1}
criterion=nn.MarginCriterion(1)

for i = 1, 1000 do
   gradUpdate(mlp, x1, x1_target, criterion, 0.01)
   gradUpdate(mlp, x2, x2_target, criterion, 0.01)
end

print(mlp:forward(x1))
print(mlp:forward(x2))

print(criterion:forward(mlp:forward(x1), x1_target))
print(criterion:forward(mlp:forward(x2), x2_target))
```

는 다음을 출력합니다:

```lua
 1.0043
[torch.Tensor of dimension 1]


-1.0061
[torch.Tensor of dimension 1]

0
0
```

다시 말해, 두 데이터 포인트의 `margin`이 `1`이고 따라서 손실이 `0`이라고 할 때,
mlp는 두 데이터 포인트를 성공적으로 분리할 수 있습니다.


<a name="nn.MultiMarginCriterion"></a>
## MultiMarginCriterion ##

```lua
criterion = nn.MultiMarginCriterion(p)
```

입력 `x`(차원 `1`인 `텐서`)와 출력 `y`(`1` <= `y` <= `x:size(1)`인 타겟 클래스 인덱스) 사이의
다 부류 분류 힌지 손실(마진 기반 손실)을 최적화하는 기준(criterion)을 만듭니다.

```lua
loss(x, y) = sum_i(max(0, 1 - (x[y] - x[i]))^p) / x:size(1)
```

여기서 `i == 1`에서 `x:size(1)`이고 `i ~= y`.
또한, 이 기준은 2차원 입력가 1차원 타겟들로도 동작합니다.

이 기준은 다음 코드의 출력 층에서 끝나는 한 모듈과 결합하여 사용될 때 분류에 특히 유용합니다: (의미 불분명?)
```lua
mlp = nn.Sequential()
mlp:add(nn.Euclidean(n, m)) -- 거리들로 구성된 벡터 출력
mlp:add(nn.MulConstant(-1)) -- distance to similarity (유사도까지의 거리, 유사한 정도?)
```


<a name="nn.MultiLabelMarginCriterion"></a>
## MultiLabelMarginCriterion ##

```lua
criterion = nn.MultiLabelMarginCriterion()
```

입력 `x`(1차원 `텐서`)와 출력 `y`(타겟 클래스 인덱스들로 구성된 1차원 `텐서`) 사이
다-부류 다-분류 힌지 손실 (마진 기반 손실)을 최적화하는 기준을 만듭니다:

```lua
loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x:size(1)
```

여기서 `i == 1`에서 `x:size(1)`, `j == 1`에서 `y:size(1)`, 
`y[j] ~= 0`, 그리고 모든 `i`와 `j`를 위해 `i ~= y[j]`.
또한 이 기준은 2차원 입력과 타겟으로도 동작할 수 있음에 유념하십시오.

`y`와 `x`의 차원은 반드시 같아야 합니다.
이 기준은 오직 첫 번째 `0`이 아닌 `y[j]` 타겟만을 고려합니다.
이 기준은 다른 샘플들이 가변적인 양(amount)으로 구성된 타겟 클래스를 가지도록 합니다:


```lua
criterion = nn.MultiLabelMarginCriterion()
input = torch.randn(2, 4)
target = torch.Tensor{{1, 3, 0, 0}, {4, 0, 0, 0}} -- 영 값들은 무시됩니다
criterion:forward(input, target)
```


<a name="nn.MSECriterion"></a>
## MSECriterion ##

```lua
criterion = nn.MSECriterion()
```

입력 `x`와 출력 `y`에 있는 `n` 요소들 사이
평균 제곱 오차를 측정하는 기준을 만듭니다.

```lua
loss(x, y) = 1/n \sum |x_i - y_i|^2 .
```

만약 `x`와 `y`가 총 `n` 요소로 구성된 `d`차원 `텐서`이면,
합 연산은 연전이 모든 요소들에 대해 동작합니다, 그리고 `n`으로 나눕니다.
그 두 `텐서`는 반드시 같은 수의 요소들을 가져야 합니다 (그러나 차원들은 다를 수도 있습니다).

만약 내부 변수 `sizeAverage`를 `false`로 설정하면, `n`으로 나누지 않도록 할 수도 있습니다:

```lua
criterion = nn.MSECriterion()
criterion.sizeAverage = false
```


<a name="nn.MultiCriterion"></a>
## MultiCriterion ##

```lua
criterion = nn.MultiCriterion()
```

다른 기준들의 가중된 합을 하나의 기준으로 리턴합니다.
기준들은 다음 메소드를 사용하여 더해집니다:

```lua
criterion:add(singleCriterion [, weight])
```

여기서 `weight`는 스칼라입니다 (기본값 1). 각 기준은 같은 `input`과 `target`에 적용됩니다.

예 :

```lua
input = torch.rand(2,10)
target = torch.IntTensor{1,8}
nll = nn.ClassNLLCriterion()
nll2 = nn.CrossEntropyCriterion()
mc = nn.MultiCriterion():add(nll, 0.5):add(nll2)
output = mc:forward(input, target)
```

<a name="nn.ParallelCriterion"></a>
## ParallelCriterion ##

```lua
criterion = nn.ParallelCriterion([repeatTarget])
```

다른 기준들의 가중된 합을 하나의 기준으로 리턴합니다.
기준들은 다음 메소드를 사용하여 더해집니다:

```lua
criterion:add(singleCriterion [, weight])
```

여기서 `weight`는 스칼라입니다 (기본값 1). 이 기준은 `input`과 `target` 테이블 하나를 기대합니다.
각 기준은 테이블 안에 있는 상응하는 `input`과 `target` 요소에 적용됩니다.
그러나 만약 `repeatTarget=true`이면, `target`은 각 기준에 반복적으로 제시됩니다 (다른 `input`으로).

예 :

```lua
input = {torch.rand(2,10), torch.randn(2,10)}
target = {torch.IntTensor{1,8}, torch.randn(2,10)}
nll = nn.ClassNLLCriterion()
mse = nn.MSECriterion()
pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)
output = pc:forward(input, target)
```


<a name="nn.SmoothL1Criterion"></a>
## SmoothL1Criterion ##

```lua
criterion = nn.SmoothL1Criterion()
```

[`AbsCriterion`](#nn.AbsCriterion)의 부드러운 버전으로 생각될 수 있는 기준을 만듭니다. 
만약 그 요소별 오차 절댓값이 1보다 작으면, 이 기준은 제곱 항을 사용합니다.
이 기준은 [`MSECriterion`](#nn.MSECriterion)보다 아웃라이어(outlier, 평균에서 많이 벗어난 값들)들에 덜 민감합니다.
그리고 어떤 경우에는 기울기 폭발을 막습니다 (예를 들어 Ross Girshick가 쓴 "Fast R-CNN" 논문을 보십시오).


```lua
                      ⎧ 0.5 * (x_i - y_i)^2, 만약 |x_i - y_i| < 1
loss(x, y) = 1/n \sum ⎨
                      ⎩ |x_i - y_i| - 0.5,   그렇지않으면
```

만약 `x`와 `y`가 총 `n`개 요소를 가진 `d`차원 텐서이면, 
그 합 연산은 여전히 모든 요소들에 대해 동작합니다, 그리고 `n`으로 나눕니다.

만약 내부 변수 `sizeAverage`를 `false`로 설정하면, `n`으로 나누지 않도록 할 수도 있습니다:

```lua
criterion = nn.SmoothL1Criterion()
criterion.sizeAverage = false
```


<a name="nn.HingeEmbeddingCriterion"></a>
## HingeEmbeddingCriterion ##

```lua
criterion = nn.HingeEmbeddingCriterion([margin])
```

주어진 1차원 벡터 입력 `x`와 레이블 `y` (`1` 또는 `-1`)으로 손실을 측정하는 기준을 만듭니다
이 기준은, 예를 들어, L1 쌍별 거리를 사용하여, 보통 두 입력이 비슷한지 안비슷한지를 측정하는 데 사용됩니다.
그리고 이 기준은 보통 비선형 임베딩들을 학습하거나 준 지도 학습을 위해 사용됩니다.


```lua
                 ⎧ x_i,                  만약 y_i ==  1
loss(x, y) = 1/n ⎨
                 ⎩ max(0, margin - x_i), 만약 y_i == -1
```

만약 `x`와 `y`가 `n`차원 텐서이면, 합 연산은 여전히 모든 요소들에 대해 동작합니다, 그리고 `n`으로 나눕니다
(만약 내부 변수 `sizeAverage`를 `false`로 설정하면, `n`으로 나누지 않도록 할 수도 있습니다).
`margin`의 기본값은 `1`입니다, 또는 생성자에서 설정될 수 있습니다.


### 예

```lua
-- 다음과 같이 우리가 관심 있어 하는 네트워크가 하나 있다고 상상해봅시다, 그것을 "p1_mlp"라 부르겠습니다.
p1_mlp = nn.Sequential(); p1_mlp:add(nn.Linear(5, 2))

-- 그러나 우리는 예제들을 밀어넣거나 서로 멀리 치우기를 원합니다.
-- 그래서 우리는 p2_mlp라 불리는 그 예제의 또다른 복사본을 만듭니다.
-- 이것은 그 set 명령어 사이에 같은 가중치들을 *공유*합니다.
-- 그러나 이것은 임시 기울기 스토리지로 구성된 그 자신의 set를 가집니다.
-- 그것이 우리가 그것을 다시 만든 이유입니다 (그 쌍의 기울기들이 서로를 지우지 않게 하려고)
p2_mlp = nn.Sequential(); p2_mlp:add(nn.Linear(5, 2))
p2_mlp:get(1).weight:set(p1_mlp:get(1).weight)
p2_mlp:get(1).bias:set(p1_mlp:get(1).bias)

-- 우리는 예제 한 쌍을 입력으로 받는 병렬 테이블 하나를 만듭니다.
-- 그 두 예제는 모두 같은 (복제된) mlp를 통과합니다.
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

-- 이제 우리는 이 병렬 테이블을 입력으로 받는 꼭대기 층 네트워크를 정의합니다.
-- 그리고 그 출력 쌍 사이 쌍별(pairwise) 거리를 계산합니다. 
mlp = nn.Sequential()
mlp:add(prl)
mlp:add(nn.PairwiseDistance(1))

-- 기준 쌍들을 모으거나 갈라 놓기 위한 기준
crit = nn.HingeEmbeddingCriterion(1)

-- 두 예제 벡터를 만듭시다.
x = torch.rand(5)
y = torch.rand(5)


-- 전형적 기울기 갱신 함수를 사용합니다.
function gradUpdate(mlp, x, y, criterion, learningRate)
local pred = mlp:forward(x)
local err = criterion:forward(pred, y)
local gradCriterion = criterion:backward(pred, y)
mlp:zeroGradParameters()
mlp:backward(x, gradCriterion)
mlp:updateParameters(learningRate)
end

-- 쌍 x와 y를 함께 넣습니다, 
-- 어떻게 print(mlp:forward({x, y})[1])로 주어진 그것들 사이 거리가 작아지는 지에 주목하십시오.
for i = 1, 10 do
   gradUpdate(mlp, {x, y}, 1, crit, 0.01)
   print(mlp:forward({x, y})[1])
end

-- 쌍 x와 y를 갈라 놓습니다, 
-- 어떻게 print(mlp:forward({x, y})[1])로 주어진 그것들 사이 거리가 커지는 지에 주목하십시오.
for i = 1, 10 do
   gradUpdate(mlp, {x, y}, -1, crit, 0.01)
   print(mlp:forward({x, y})[1])
end
```


<a name="nn.L1HingeEmbeddingCriterion"></a>
## L1HingeEmbeddingCriterion ##

```lua
criterion = nn.L1HingeEmbeddingCriterion([margin])
```

주어진 입력 `x` = `{x1, x2}`, 두 텐서로 구성된 테이블, 그리고 레이블 `y`(`1` 또는 `-1`)으로
손실을 측정하는 기준 하나를 만듭니다.
이 기준은 L1 거리를 사용하여 두 입력이 비슷한지 안비슷한지를 측정하는 데 사용됩니다.
그리고 이 기준은 보통 비선형 임베딩들을 학습하거나 준 지도 학습을 위해 사용됩니다.

```lua
             ⎧ ||x1 - x2||_1,                  만약 y ==  1
loss(x, y) = ⎨
             ⎩ max(0, margin - ||x1 - x2||_1), 만약 y == -1
```

`margin`의 기본값은 `1`입니다. 또한, `margin`은 생성자로 설정될 수 있습니다.

<a name="nn.CosineEmbeddingCriterion"></a>
## CosineEmbeddingCriterion ##

```lua
criterion = nn.CosineEmbeddingCriterion([margin])
```

주어진 입력 `x` = `{x1, x2}`, 두 `텐서`로 구성된 테이블, 그리고 값 1 또는 -1인을 가진 `텐서` 레이블 `y`으로 
손실을 측정하는 기준을 만듭니다.
코사인 거리를 사용하여, 이 함수는 두 입력이 비슷한지 안비슷한지 측정하는 데 사용됩니다.
그리고 이 함수는 비선형 임베딩 또는 준 지도(semi-supervised) 학습을 위해 사용됩니다.

`margin`은 `-1` 부터 `1` 까지의 숫자여야 합니다. `0` 에서 `0.5`가 제안됩니다.
`Forward`와 `Backward`는 둘 중 하나만 사용되어야 합니다.
만약 `margin`이 없으면, 기본값은 `0`입니다.

각 샘플을 위한 손실 함수는:

```lua
             ⎧ 1 - cos(x1, x2),              만약 y ==  1
loss(x, y) = ⎨
             ⎩ max(0, cos(x1, x2) - margin), 만약 y == -1
```

묶음(batch)으로 된 입력들을 위해, 만약 내부 변수 `sizeAverage`가 `true`이면,
손실 함수는 그 배치 샘플들에 대한 평균을 냅니다;
만약 `sizeAverage`가 `false`이면, 손실 함수는 배치 샘플들에 대한 합을 계산합니다.
`sizeAverage`의 기본값은 `true`입니다.


<a name="nn.MarginRankingCriterion"></a>
## MarginRankingCriterion ##

```lua
criterion = nn.MarginRankingCriterion(margin)
```

주어진 입력 `x` = `{x1, x2}`, (오직 스칼라로만 구성된) 크기가 1인 두 `텐서`로 구성된 테이블, 
그리고 레이블 `y`(`1` 또는 `-1`)로 
손실을 측정하는 기준을 만듭니다.

만약 `y == 1`이면, 이 함수는 첫 번째 입력이 반드시 두 번째 입력보다 더 높게 순위 매겨져야 한다고 가정됩니다,
그리고 `y == -1`이면, 그 반대로 가정됩니다.

손실 함수는:

```lua
loss(x, y) = max(0, -y * (x[1] - x[2]) + margin)
```

### 예

```lua
p1_mlp = nn.Linear(5, 2)
p2_mlp = p1_mlp:clone('weight', 'bias', 'gradWeight', 'gradBias')

prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

mlp1 = nn.Sequential()
mlp1:add(prl)
mlp1:add(nn.DotProduct())

mlp2 = mlp1:clone('weight', 'bias', 'gradWeight', 'gradBias')

mlpa = nn.Sequential()
prla = nn.ParallelTable()
prla:add(mlp1)
prla:add(mlp2)
mlpa:add(prla)

crit = nn.MarginRankingCriterion(0.1)

x=torch.randn(5)
y=torch.randn(5)
z=torch.randn(5)

-- 전형적이고 일반적인 기울기 갱신 함수를 사용합니다
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

for i = 1, 100 do
   gradUpdate(mlpa, {{x, y}, {x, z}}, 1, crit, 0.01)
   if true then
      o1 = mlp1:forward{x, y}[1]
      o2 = mlp2:forward{x, z}[1]
      o = crit:forward(mlpa:forward{{x, y}, {x, z}}, 1)
      print(o1, o2, o)
   end
end

print "--"

for i = 1, 100 do
   gradUpdate(mlpa, {{x, y}, {x, z}}, -1, crit, 0.01)
   if true then
      o1 = mlp1:forward{x, y}[1]
      o2 = mlp2:forward{x, z}[1]
      o = crit:forward(mlpa:forward{{x, y}, {x, z}}, -1)
      print(o1, o2, o)
   end
end
```
