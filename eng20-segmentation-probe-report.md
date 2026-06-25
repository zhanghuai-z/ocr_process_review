# Eng20 / EngCut 切分能力实验

本文回答：Eng20 是否能自己切英文框，以及能否用它替代当前 Hanwang 字符框定位。

## 实验输出

```text
scripts/experiment_eng20_segmentation.py
debug/eng20_segmentation_probe/eng20_segmentation_probe.md
debug/eng20_segmentation_probe/eng20_segmentation_probe.json
debug/eng20_segmentation_probe/*.overlay.png
debug/eng20_line_box_echo/raw_eng20_line_box_echo_contact.png
debug/eng20_line_box_echo/raw_eng20_probe/
```

测试了两种接口：

```text
recogline_engstr
recogimg_engstr
```

## 结论

Eng20 有字符 bbox 输出能力。

在纯英文 crop 上表现正常：

```text
Specialization -> Specialization
PE/VC          -> PE/VC
```

并且每个字符都有 crop-local bbox。

在纯净混合中文 line 上，Eng20 可以召回一部分拉丁 token，并返回字符框：

```text
Specialization / SPE 所在整行
  -> ~J~{~~t,2011),~:h\u0080~~ktb~~~~~Sc(Specialization,SPE)fti~a#fb~~~%~t

TFP / Li / Lyv 所在整行
  -> ~J'~3F~iBTX~&~Ef{Jft~izSt4:gg~i+,~-rfiTFPfbi+%~-5LialLyv(2021)~+~YL

PE/VC / TFP 所在整行
  -> tj<JPE/VC~k,~fffeI~~-i~j;3t~-j~ii-~~kTFPa<J~7R~~~~ibBA~o1~~(3)3iUb
```

这说明 line 级 Eng20 不是完全不可用。它能在整行里识别到 `Specialization`、`SPE`、`TFP`、`PE/VC` 这类目标词，并给出字符 bbox。

但它也会把大量中文笔画识别成 `~`、数字、随机拉丁字母和符号。因此它不能独立决定“哪里是英文”，只能在 Paddle/VL 已经给出目标 token 的前提下，用作候选匹配源。

## 判断

```text
Eng20 可以作为：
  已有 crop -> 英文识别 + 字符 bbox 输出
  纯净 line -> 在 Paddle/VL token 约束下召回英文候选框

Eng20 不适合作为：
  混合中文整行 -> 无约束自动发现英文位置
```

所以当前主线判断不变：

```text
定位：
  Paddle/VL 真值文本 + Hanwang 字符框 + 文本对齐

识别验证：
  小 crop -> Eng20
```

可以把整行交给 Eng20 做候选召回，但不能无约束采纳它输出的随机拉丁串。

## line 级匹配假设

有一个可用但受限的思路：

```text
输入：
  PP-OCRv5 提供的一整条 line crop

Eng20 输出：
  它认为像英文的字符序列 + 每个字符 bbox

匹配方式：
  用 Paddle/VL 已知的英文 token 去 Eng20 的 line 结果里做精确/近似匹配。
  如果匹配成功，就把对应字符 bbox 合并成英文 token bbox。
```

这个思路不仅理论上成立，纯净 line 实测也能召回目标 token：

```text
Specialization / SPE 所在 line
  expected: Specialization, SPE
  Eng20:    ~J~{~~t,2011),~:h\u0080~~ktb~~~~~Sc(Specialization,SPE)fti~a#fb~~~%~t
  结果：召回 Specialization 和 SPE，但周围有大量噪声。

TFP / Li / Lyv 所在 line
  expected: TFP, Li, Lyv
  Eng20:    ~J'~3F~iBTX~&~Ef{Jft~izSt4:gg~i+,~-rfiTFPfbi+%~-5LialLyv(2021)~+~YL
  结果：召回 TFP、Li、Lyv，但 Li/Lyv 中间边界可能粘连。

PE/VC / TFP 所在 line
  expected: PE/VC, TFP
  Eng20:    tj<JPE/VC~k,~fffeI~~-i~j;3t~-j~ii-~~kTFPa<J~7R~~~~ibBA~o1~~(3)3iUb
  结果：召回 PE/VC 和 TFP，但同样混有噪声。
```

所以 line 级 Eng20 更适合做“弱候选”：

```text
可采用：
  Eng20 line 结果中出现了 Paddle/VL 目标 token 的 exact match 或高置信近似匹配
  -> 可作为 bbox 候选或 review 辅助

不可采用：
  Eng20 line 输出里的随机拉丁串
  -> 不能反推它一定是正文里的英文 token
```

注意：早期 line 实验误用了已经画过 Hanwang 回显框的 line 图片，导致 Eng20 输入被蓝/红框污染；这组结果不再作为判断依据。当前判断以 `debug/eng20_line_box_echo/raw_lines/` 里的纯净 line crop 为准。

## 脏 crop 同场景测试

输出目录：

```text
debug/eng20_contaminated_crop_probe/
```

这些样例不是整行，而是当前链路里已经混入相邻像素或相邻字符的 crop。结论是：Eng20/EngCut 不会自动清理 crop 边界，它会把 crop 里看到的相邻字符也一起识别出来。

```text
120171 GDP crop
  expected target: GDP
  Eng20: rGDP
  说明：左侧混入的 r 被一起读出。

120192 Age crop
  expected target: Age
  Eng20: (Age)
  说明：括号没有被自动剥离。

120183 zju.edu.cn crop
  expected target: zju.edu.cn
  Eng20: )zju.edu.cn
  说明：前置右括号被一起读出。

120186 PE/VC crop
  expected target: PE/VC
  Eng20: PE!VC
  说明：符号 `/` 仍可能被误识别成 `!`。

120186 过宽 PE/VC crop
  expected target: PE/VC
  recogline_engstr: VC~~i~~~f9LJ~K~la.jF~iZft9~2k~~19(Carpenter81Petersen,2002)o~l~~&
  recogimg_engstr:  4k\IV~~~~fjf<J6T~~~,EicEt9J~~iBl~,Fi~K~~JJ&:~aF~it,~ffi~~PE/VC~~i~~~f9LJ~K~la.jF~iZft9~2k~~19(Carpenter81Petersen,2002)o~l~~&
  说明：crop 一旦跨入大段正文，Eng20 会退化成噪声识别；recogimg_engstr 可能包含 PE/VC，但不是一个可直接使用的干净目标。
```

因此，Eng20 不能作为“脏 crop 修复器”。如果上游 crop 已经混入其他像素，Eng20 大概率会把污染内容读进去，或者在过宽 crop 上产生大量噪声。

## 风险样例

120186 的 `PE/VC` crop：

```text
expected: PE/VC
Eng20:    PE!VC
```

这说明即使 crop 已经很小，Eng20 对 `/` 这类符号仍可能误识别，需要 review flag，不能自动覆盖正文。

## 对主链路的影响

当前应该修上游 crop 纯度，而不是指望 Eng20 清理：

```text
1. Latin crop 边界要从 Hanwang 字符框/PP-OCRv5 行框中收紧。
2. crop 过宽、过高、跨行时直接标记 review，不自动采用 Eng20。
3. 前后括号、标点这类边界污染可以做显示层提示，但不能无条件删除。
4. Eng20 适合作为小 crop 的英文复核和字符 bbox 来源，不适合作为目标英文定位器。
```
