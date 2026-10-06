# Torchstaar独立原R验证

验证器只读取同一任务的原R与Torchstaar原生输出，不运行关联或GPU。生产分析、Python/CLI、输入格式、完整795任务与15mask见 [Torchstaar指南](../docs/torchstaar.md)。当前真实完整结果为795任务/19文件、strict null及显著联合精度通过，关联进程墙钟296.196098秒；独立CPU R比较127.953秒另计。

## 输入与输出

输入为同一冻结模型/样本/变异/注释/完整目录的Rdata或RDS，必须使用对应basename和原保存对象顺序。原R reference只读，candidate是Python写出的原生文件。输出JSON为匿名结构/数值计数、失败数与最大误差；个体状态、任务身份与实际结果表始终留在私有目录。

## 调用

```bash
# 全部原生typeof/属性/顺序/标识/NULL。
Rscript validation/compare_structure.R \
  private/reference/result.Rdata private/candidate/result.Rdata private/structure.json
# 严格null和原合法空结果的非P数值比较。
Rscript validation/compare_nonp.R \
  private/reference/obj_nullmodel.Rdata private/candidate/obj_nullmodel.Rdata private/null.json
# 原完整严格数值控制；关联非P误差为诊断，不能替换主显著联合gate。
Rscript validation/compare_native_outputs.R \
  private/reference/result.Rdata private/candidate/result.Rdata
```

| 脚本参数 | 意义 |
|---|---|
| reference / candidate | 前两项，顺序固定，同批次原R/Torchstaar Rdata或RDS。 |
| report | structure/nonp第三项，匿名JSON输出路径。 |
| native comparator atol / rtol | 可选第三/第四项；默认1e-10/1e-7，double阈值为 `absdiff<=atol+rtol*abs(reference)`。当前strict null固定原门槛。 |

结构比较要求类型、class、维度、名称/属性顺序、factor levels、row.names、S4 slots、NA/NaN/Inf与NULL一致；整数/字符串保持精确。严格null必须独立通过全部非P数值检查，不能因没有关联P免验。18关联文件含15个P文件和3个原合法空文件；只有绑定原reference预检证实双方无P、结构及非P严格比较通过时，空文件P门槛才是不适用，原raw validator false保留。

完整主P规则、下溢/native log、分层诊断与脚本参数见 [logP验证](README_logp.md)。单个文件比较不能代替完整目录/795任务/19文件及全部mask覆盖证明。验证环境需要R、Matrix和jsonlite；生产环境不因此增加R依赖。

## 当前验证与参考

全478,082 P有效可比较，显著联合24,713/0超限，最大logP差0.000357971421594216；strict null341,221数值单元格/0超限。全部P与关联非P严格误差保留诊断false。近期版本/资源/计时统一见 [真实结果](../docs/torchstaar.md#5-最新真实精度与耗时)和 [匿名JSON](../benchmarks/torchstaar_chr21_2026-10-06.json)。

原函数来源为锁定的STAAR/STAARpipeline与PheWAS源码，链接和方法文献见 [主指南参考](../docs/torchstaar.md#7-原实现许可与参考文献)。验证脚本保留原实现规则，本次文档整理不修改比较器。
