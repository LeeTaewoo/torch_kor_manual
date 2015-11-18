# 최적화 패키지

이 패키지는 [Torch](https://github.com/torch/torch7/blob/master/README.md)를 위한 몇 개의 최적화 루틴들을 담고 있습니다.
각 최적화 알고리즘은 같은 인터페이스에 기반을 둡니다:

```lua
x*, {f}, ... = optim.method(func, x, state)
```

여기서:

* `func`: 사용자가 정의한 함수안에 정의된 함수(closure)입니다. 다음 API를 따릅니다: `f, df/dx = func(x)`
* `x`: 현재 파라미터 벡터 (1차원 `torch.Tensor`)
* `state`: 파라미터들로 구성된 테이블 하나, 그리고 알고리즘 의존적 상태 변수들
* `x*`: `f, x* = argmin_x f(x)`를 최소화하는 새 파라미터 벡터
* `{f}`: 모든 f 값들의 테이블 하나, 그것들이 평가된 순서대로 (SGD 같은 몇몇 단순한 알고리즘의 경우, `#f == 1`)

## 중요한 노트

상태 테이블은 그 알고리즘의 상태를 들고 있기 위해 사용됩니다.
그 테이블은 보통 사용자에 의해 한 번만 초기화됩니다. 그리고 optim 함수로 블랙 박스처럼 전달됩니다.
예:

```lua
state = {
   learningRate = 1e-3,
   momentum = 0.5
}

for i,sample in ipairs(training_samples) do
    local func = function(x)
       -- 평가(eval) 함수 정의
       return f,df_dx
    end
    optim.sgd(func,x,state)
end
```
