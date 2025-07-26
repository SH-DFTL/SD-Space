\## 环境

\## 数据集

```python

\#数据以天为单位存放在nc格式的文件中，打印内容如下：

xarray.Dataset> Size: 6kB

Dimensions:  (time: 24, height: 55)

Coordinates:

&nbsp;   lat      float64 8B ...

&nbsp;   lon      float64 8B ...

&nbsp; \* time     (time) datetime64\[ns] 192B 2002-01-01 ... 2002-01-01T23:00:00

&nbsp; \* height   (height) float32 220B 60.0 70.0 80.0 90.0 ... 580.0 590.0 600.0

Data variables:

&nbsp;   ELECDEN  (time, height) float32 5kB ...

&nbsp;   ap       (time) float64 192B ...

&nbsp;   f107     (time) float64 192B ...

&nbsp;   f107a    (time) float64 192B ...

```

