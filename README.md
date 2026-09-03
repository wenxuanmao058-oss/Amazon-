# Amazon 电子配件品类运营分析

基于 Amazon.in（印度站）真实评论数据，从**选品 → Listing 优化 → 广告预算**形成一套完整的运营分析闭环。

## 项目说明

- **数据来源**：Amazon.in 电子配件商品与评论（Kaggle 数据集 `talalhakem/amazon`，约 1,464 个商品、12,154 条评论）。
- **分析目标**：帮运营人员判断"这个品类值不值得进、Listing 怎么优化、广告预算怎么控"。
- **技术栈**：Python（pandas / numpy / matplotlib）。

## 文件结构

```
.
├── amazon_analysis.py   主分析脚本（清洗→选品→痛点→结论，并出图）
├── data_prepare.py       数据清洗与拆分（商品表 / 评论长表）
├── amazon analysis.ipynb 交互式运行 Notebook

```

## 运行方式

1. 安装依赖
   ```
   pip install pandas numpy matplotlib
   ```
2. 把 `amazon.csv` 放到本目录（与脚本同级）
3. 运行主脚本
   ```
   python amazon_analysis.py
   ```

## 数据说明

- 原始数据 `amazon.csv` 来自 Kaggle：[talalhakem/amazon](https://www.kaggle.com/datasets/talalhakem/amazon)
- 需自行下载后放到项目根目录，脚本才能读取。

## 分析内容

### 1. 商品运营洞察
- **品类供给集中度**：USBCables 等品类商品数最多
- **需求热度**：按平均评论量（rating_count 代理）看流量空间
- **价格带 / 折扣**：折后价中位数 ₹799，平均折扣约 47.7%
- **口碑结构**：按平均评分看竞品口碑
- **综合选品候选**：结合商品数、热度、口碑多维排序，锁定 **HDMICables**

### 2. 评论痛点挖掘
- 对 12,154 条评论做**情感分析**（词典法），定位负向评论
- 归纳核心痛点：**做工质量（Build Quality）/ 充电（Charging）/ 配送售后（Delivery & Service）**
- 并将做工质量细分为：线材与物理损坏、材质做工、屏幕显示、售后服务

### 3. 运营结论
- 优先进入 **HDMICables**（商品数、需求热度、口碑三者均衡）
- 主推折后价 ₹799 附近产品
- 优先解决 **做工质量** 问题
- 用户认可的长板（耐用性、快充、易用性）前置为 Listing 卖点，规避过度承诺

## 免责声明

- 情感分析基于**规则词典**，属粗粒度判断，不代表绝对情感占比。
- 数据为印度站（Amazon.in），价格单位为 ₹（卢比），结论迁移到其他站点需重新校准。
- 本仓库用于学习与求职展示，不构成商业建议。
