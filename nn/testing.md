## 테스팅 ##
만약 자신만의 모듈을 구현하고 싶은 사람들을 위해, 우리는
[torch.Tester](https://github.com/torch/torch7/blob/master/doc/tester.md) 클래스와 함께 
그 클래스의 미분들을 시험하기 위해 `nn.Jacobian` 클래스를 사용할 것을 제안합니다. 
`nn` 패키지의 소스들에는 그런 시험들을 위한 충분히 많은 예제들이 포함되어 있습니다.
