r"""
亚马逊评论 + 商品运营分析脚本

统计口径说明：
- 商品表和评论表由 data_prepare.py 从原始 amazon.csv 在内存中生成，不落盘。
- rating 是商品整体评分，不是每条评论的单独评分。
- rating_count 是商品被评论的次数，用来代理商品热度/销量。
- category 使用 "|" 分层，取最后一段作为叶子品类。
- 情感分析基于英文正负词和短语词典，不依赖单条评论星级。
- 痛点挖掘只统计负向评论，按关键词归类；一条评论可以命中多个痛点。
- 品类需求热度和口碑排名只保留商品数 >= 20 的品类，避免小样本误判。
"""

import importlib
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_prepare

importlib.reload(data_prepare)


# 品类结论至少需要 20 个商品样本，避免小样本品类被误判为市场机会。
MIN_PRODUCTS = 20


# 英文停用词：词频统计时先去掉这些高频无业务含义词。
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "don't", "should", "now", "i",
    "me", "my", "we", "our", "you", "your", "it", "its", "they",
    "them", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "could", "should",
    "may", "might", "must", "also", "very", "really", "got", "get",
}


# 正向词：表示用户认可、满意、推荐。
POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "amazing", "perfect", "nice",
    "best", "better", "fine", "fast", "quick", "easy", "value", "worth",
    "satisfied", "happy", "recommend", "quality", "durable", "sturdy",
    "working", "works", "love", "loved", "beautiful", "premium", "solid",
    "reliable", "comfortable", "impressive", "perfectly", "super", "useful",
}


# 负向词：表示用户不满意、质量差、退货等。
NEGATIVE_WORDS = {
    "bad", "poor", "worst", "terrible", "disappointed", "disappointing",
    "waste", "useless", "defective", "broken", "damaged", "faulty",
    "issue", "problem", "slow", "unreliable", "fake", "cheap", "badly",
    "not", "no", "never", "hard", "difficult", "horrible", "awful",
    "unhappy", "regret", "fail", "failed", "failure", "stopped",
    "returned", "return", "replace", "replacement", "refund", "cracked",
}


# 负向短语：比单字更能反映真实语义，例如 "not working" 要整体识别。
NEGATIVE_PHRASES = [
    "not good", "not great", "not working", "not charge", "not charging",
    "does not work", "doesn't work", "does not charge", "doesn't charge",
    "stopped working", "not satisfied", "not happy", "not recommend",
    "no sound", "no power", "no issue", "no complaints", "no complain",
    "not compatible", "not durable", "waste of money", "not worth",
    "poor quality", "low quality", "very poor", "very bad", "too slow",
    "slow charging", "charging issue", "battery drain", "drains quickly",
    "overheat", "heating issue", "getting hot", "gets hot", "very hot",
    "screen issue", "display issue", "sound issue", "audio issue",
    "connectivity issue", "connection issue", "frequently disconnects",
    "keep disconnecting", "randomly disconnects", "stopped", "broken",
    "cracked", "damaged", "defective", "not work", "won't charge",
    "not working properly", "not fit", "not good quality", "not up to",
    "not accurate", "not clear", "not loud", "not bright", "not crisp",
    "not stable", "not smooth", "not easy", "not compatible",
]


# 正向短语：常见用户好评表达。
POSITIVE_PHRASES = [
    "very good", "very nice", "great product", "good product", "works well",
    "working fine", "works fine", "working great", "works great",
    "value for money", "worth the money", "highly recommend", "good quality",
    "great quality", "best quality", "very fast", "fast charging",
    "easy to use", "easy to install", "perfect fit", "solid build",
    "good sound", "great sound", "good picture", "great picture",
    "battery life is good", "good battery", "long battery", "very durable",
]


# 痛点关键词：只对负向评论做匹配，用来归纳用户抱怨的类别。
PAIN_PATTERNS = {
    "Charging / Battery": [
        "charge", "charging", "charger", "battery", "power", "drain",
        "adapter", "cable",
    ],
    "Build Quality": [
        "quality", "durab", "break", "broke", "broken", "crack", "damag",
        "defect", "cheap", "material", "plastic", "build", "flimsy",
        "waste", "poor", "stopped working",
    ],
    "Compatibility / Connection": [
        "compatib", "connect", "connection", "pair", "pairing", "bluetooth",
        "android", "ios", "iphone", "ipad", "type c", "lightning", "usb",
    ],
    "Sound / Display": [
        "sound", "audio", "bass", "volume", "mic", "noise", "screen",
        "display", "picture", "resolution", "bright", "panel", "video",
    ],
    "Heating": [
        "heat", "hot", "overheat", "warm", "burn", "temperature",
    ],
    "Delivery / Service": [
        "delivery", "ship", "arrive", "arrived", "packag", "replace",
        "return", "refund", "customer", "seller", "warranty", "late",
        "missing", "damaged during",
    ],
    "Price / Value": [
        "price", "value", "worth", "money", "cost", "expensive", "cheap",
    ],
    "Installation / Setup": [
        "install", "setup", "set up", "fitting", "mount", "configure",
    ],
}


# Build Quality 大类的细拆，用来进一步解释用户到底在抱怨什么。
BUILD_QUALITY_DETAILS = {
    "Cable / Physical Damage": [
        "cable", "wire", "broke", "broken", "crack", "damag", "fray",
        "stopped working", "not working",
    ],
    "Display / Screen Issue": [
        "display", "screen", "picture", "panel", "pixel",
    ],
    "Material / Build": [
        "quality", "cheap", "plastic", "flimsy", "build", "material",
    ],
    "After-sales / Service": [
        "service", "warranty", "replace", "return", "refund", "customer",
    ],
}


# 报告里的痛点中文说明，只翻译痛点标签，不翻译产品和品类名。
PAIN_LABELS_CN = {
    "Charging / Battery": "Charging / Battery（充电 / 电池）",
    "Build Quality": "Build Quality（做工质量）",
    "Compatibility / Connection": "Compatibility / Connection（兼容性 / 连接）",
    "Sound / Display": "Sound / Display（音质 / 显示）",
    "Heating": "Heating（发热）",
    "Delivery / Service": "Delivery / Service（配送 / 售后）",
    "Price / Value": "Price / Value（价格 / 性价比）",
    "Installation / Setup": "Installation / Setup（安装 / 设置）",
}


BUILD_QUALITY_LABELS_CN = {
    "Cable / Physical Damage": "Cable / Physical Damage（线材 / 物理损坏）",
    "Display / Screen Issue": "Display / Screen Issue（屏幕 / 显示问题）",
    "Material / Build": "Material / Build（材质 / 做工）",
    "After-sales / Service": "After-sales / Service（售后服务）",
}


def to_number(value) -> float:
    """把价格、百分比、评论数统一转成数字。"""
    if pd.isna(value):
        return np.nan
    text = re.sub(r"[₹,%,\s]", "", str(value))
    try:
        return float(text)
    except ValueError:
        return np.nan


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """清洗商品字段，只保留价格、评分、折扣有效的商品。"""
    df = df.copy()

    # 价格和折扣只需要去货币符号、百分号和逗号。
    for column in ["discounted_price", "actual_price", "discount_percentage"]:
        df[column] = df[column].map(to_number).astype(float)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = df["rating_count"].map(to_number).astype(float)

    # 把完整品类路径拆成一级品类和叶子品类。
    df["top_category"] = df["category"].str.split("|").str[0]
    df["leaf_category"] = df["category"].str.split("|").str[-1]

    valid = (
        (df["discounted_price"] > 0)
        & (df["actual_price"] > 0)
        & (df["rating"] > 0)
        & (df["rating"] <= 5)
        & (df["discount_percentage"] >= 0)
        & (df["discount_percentage"] <= 100)
    )
    return df[valid].copy()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """从原始 CSV 内存读取，并完成基础字段处理。"""
    raw_products, reviews = data_prepare.load_raw_data()
    products = clean_products(raw_products)

    reviews["review_content"] = reviews["review_content"].fillna("")
    reviews["review_title"] = reviews["review_title"].fillna("")
    reviews["top_category"] = reviews["category"].str.split("|").str[0]
    reviews["leaf_category"] = reviews["category"].str.split("|").str[-1]
    return products, reviews


def normalize_text(text: str) -> str:
    """英文评论统一小写，并把非字母数字字符变成空格。"""
    text = re.sub(r"[^a-z0-9' ]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def sentiment_score(text: str) -> float:
    """按正向/负向词和短语计算情感分。"""
    text = normalize_text(text)
    positive = sum(text.count(phrase) for phrase in POSITIVE_PHRASES)
    negative = sum(text.count(phrase) for phrase in NEGATIVE_PHRASES)
    positive += sum(text.split().count(word) for word in POSITIVE_WORDS)
    negative += sum(text.split().count(word) for word in NEGATIVE_WORDS)
    return positive - negative


def classify_sentiment(score: float) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def add_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()
    reviews["sentiment_score"] = reviews["review_content"].map(sentiment_score)
    reviews["sentiment"] = reviews["sentiment_score"].map(classify_sentiment)
    return reviews


def match_pain_points(text: str) -> list[str]:
    """返回一条评论命中的痛点类别。"""
    text = text.lower()
    return [
        category
        for category, keywords in PAIN_PATTERNS.items()
        if any(keyword in text for keyword in keywords)
    ]


def count_pain_points(reviews: pd.DataFrame) -> Counter:
    """统计负向评论中的痛点次数。"""
    counter: Counter[str] = Counter()
    negative_texts = reviews.loc[reviews["sentiment"] == "negative", "review_content"]
    for text in negative_texts:
        counter.update(match_pain_points(text))
    return counter


def match_build_quality_details(text: str) -> list[str]:
    """对 Build Quality 负向评论再做一层细分。"""
    text = text.lower()
    return [
        category
        for category, keywords in BUILD_QUALITY_DETAILS.items()
        if any(keyword in text for keyword in keywords)
    ]


def count_build_quality_details(reviews: pd.DataFrame) -> Counter:
    """统计 Build Quality 大类下的细分标签。"""
    counter: Counter[str] = Counter()
    negative = reviews[reviews["sentiment"] == "negative"]
    build_quality_reviews = negative[
        negative["review_content"].map(
            lambda text: "Build Quality" in match_pain_points(text)
        )
    ]
    for text in build_quality_reviews["review_content"]:
        counter.update(match_build_quality_details(text))
    return counter


def top_terms(reviews: pd.DataFrame, n: int = 20) -> list[tuple[str, int]]:
    """统计评论高频词，去掉停用词和过短词。"""
    counter: Counter[str] = Counter()
    for text in reviews["review_content"].dropna():
        words = re.findall(r"[a-z']+", text.lower())
        counter.update(word for word in words if word not in STOPWORDS and len(word) > 2)
    return counter.most_common(n)


def is_notebook() -> bool:
    """判断当前是否在 Jupyter Notebook 中运行。"""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def show_chart(fig) -> None:
    fig.tight_layout()
    if is_notebook():
        plt.show()
    plt.close(fig)


def plot_category_counts(products: pd.DataFrame) -> None:
    counts = products["leaf_category"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    counts.plot.barh(ax=ax, color="#2f6f9f")
    ax.set_title("Top 10 Product Categories by Listing Count")
    ax.set_xlabel("Listings")
    ax.set_ylabel("Leaf Category")
    ax.grid(axis="x", alpha=0.2)
    show_chart(fig)


def plot_price_distribution(products: pd.DataFrame) -> None:
    # 用 98 分位数封顶，避免极端高价把主要价格带压扁。
    price_upper = products["discounted_price"].quantile(0.98)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(
        products.loc[products["discounted_price"] <= price_upper, "discounted_price"],
        bins=35,
        color="#2f6f9f",
        edgecolor="white",
    )
    ax.set_title("Discounted Price Distribution (98th Percentile Cap)")
    ax.set_xlabel("Discounted Price (INR)")
    ax.set_ylabel("Products")
    ax.grid(axis="y", alpha=0.2)
    show_chart(fig)


def plot_discount_analysis(products: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(
        products["discount_percentage"],
        bins=30,
        color="#3f8e6b",
        edgecolor="white",
    )
    axes[0].set_title("Discount Percentage Distribution")
    axes[0].set_xlabel("Discount (%)")
    axes[0].set_ylabel("Products")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].scatter(
        products["discount_percentage"],
        products["rating"],
        alpha=0.25,
        color="#3f8e6b",
    )
    axes[1].set_title("Discount vs Product Rating")
    axes[1].set_xlabel("Discount (%)")
    axes[1].set_ylabel("Rating")
    axes[1].grid(alpha=0.2)
    show_chart(fig)


def plot_rating_analysis(products: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(
        products["rating"],
        bins=30,
        color="#c47b36",
        edgecolor="white",
    )
    axes[0].set_title("Product Rating Distribution")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Products")
    axes[0].grid(axis="y", alpha=0.2)

    category_rating = products.groupby("leaf_category")["rating"].mean()
    category_rating.sort_values(ascending=False).head(10).plot.barh(
        ax=axes[1], color="#c47b36"
    )
    axes[1].set_title("Top 10 Categories by Average Rating")
    axes[1].set_xlabel("Average Rating")
    axes[1].set_xlim(0, 5)
    axes[1].grid(axis="x", alpha=0.2)
    show_chart(fig)


def plot_popularity(products: pd.DataFrame) -> None:
    colors = {
        "Electronics": "#2f6f9f",
        "Computers&Accessories": "#3f8e6b",
        "Home&Kitchen": "#c47b36",
        "OfficeProducts": "#8a6fa8",
    }
    point_colors = products["top_category"].map(colors).fillna("#777777")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(
        products["rating"],
        np.log10(products["rating_count"].clip(lower=1)),
        c=point_colors,
        alpha=0.32,
    )
    ax.set_title("Product Rating vs Review Count (Popularity Proxy)")
    ax.set_xlabel("Rating")
    ax.set_ylabel("log10(Review Count)")
    ax.grid(alpha=0.2)
    show_chart(fig)


def plot_sentiment(reviews: pd.DataFrame) -> None:
    counts = reviews["sentiment"].value_counts().reindex(
        ["positive", "neutral", "negative"]
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    counts.plot.bar(
        ax=ax,
        color=["#3f8e6b", "#c47b36", "#b14d43"],
        rot=0,
    )
    ax.set_title("Review Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Reviews")
    ax.grid(axis="y", alpha=0.2)
    show_chart(fig)


def plot_pain_points(reviews: pd.DataFrame) -> None:
    pain_counts = pd.Series(count_pain_points(reviews))
    pain_counts = pain_counts.sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    pain_counts.sort_values().plot.barh(ax=ax, color="#b14d43")
    ax.set_title("Top Pain Points in Negative Reviews")
    ax.set_xlabel("Mentions")
    ax.set_ylabel("Pain Point")
    ax.grid(axis="x", alpha=0.2)
    show_chart(fig)


def plot_top_terms(reviews: pd.DataFrame) -> None:
    terms = top_terms(reviews, 20)
    words = [word for word, _ in terms]
    counts = [count for _, count in terms]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(words[::-1], counts[::-1], color="#2f6f9f")
    ax.set_title("Most Common Words in Reviews")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    ax.grid(axis="x", alpha=0.2)
    show_chart(fig)


def make_charts(products: pd.DataFrame, reviews: pd.DataFrame) -> None:
    """按模块生成所有图表。"""
    plot_category_counts(products)
    plot_price_distribution(products)
    plot_discount_analysis(products)
    plot_rating_analysis(products)
    plot_popularity(products)
    plot_sentiment(reviews)
    plot_pain_points(reviews)
    plot_top_terms(reviews)


def category_rank(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """按某个指标给叶子品类排名，商品数不足 MIN_PRODUCTS 的不参与。"""
    stats = df.groupby("leaf_category")[column].agg(["mean", "count"])
    stats = stats[stats["count"] >= MIN_PRODUCTS]
    return stats.sort_values("mean", ascending=False).head(5)


def opportunity_category(products: pd.DataFrame) -> pd.DataFrame:
    """综合需求热度和口碑排序，避免只看平均评论数或平均评分。"""
    stats = products.groupby("leaf_category").agg(
        product_count=("product_id", "count"),
        avg_reviews=("rating_count", "mean"),
        avg_rating=("rating", "mean"),
    )
    stats = stats[stats["product_count"] >= MIN_PRODUCTS]
    stats["demand_rank"] = stats["avg_reviews"].rank(ascending=False)
    stats["rating_rank"] = stats["avg_rating"].rank(ascending=False)
    stats["score"] = stats["demand_rank"] + stats["rating_rank"]
    return stats.sort_values("score").head(5)


def build_report(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    terms: list[tuple[str, int]],
) -> str:
    """汇总商品指标、评论情感和痛点，生成 Markdown 报告。"""
    top_counts = products["leaf_category"].value_counts().head(5)
    demand_rank = category_rank(products, "rating_count")
    rating_rank = category_rank(products, "rating")
    opportunity_rank = opportunity_category(products)
    sentiment_counts = reviews["sentiment"].value_counts()
    pain_counts = count_pain_points(reviews)
    build_quality_counts = count_build_quality_details(reviews)
    pain_rows = pain_counts.most_common(5)

    count_rows = "\n".join(
        f"| {category} | {count} |" for category, count in top_counts.items()
    )
    demand_rows = "\n".join(
        f"| {category} | {row['mean']:,.0f} | {int(row['count'])} |"
        for category, row in demand_rank.iterrows()
    )
    rating_rows = "\n".join(
        f"| {category} | {row['mean']:.2f} | {int(row['count'])} |"
        for category, row in rating_rank.iterrows()
    )
    opportunity_rows = "\n".join(
        f"| {category} | {row['avg_reviews']:,.0f} | {row['avg_rating']:.2f} | {int(row['product_count'])} |"
        for category, row in opportunity_rank.iterrows()
    )
    pain_rows_md = "\n".join(
        f"| {PAIN_LABELS_CN.get(category, category)} | {count} |"
        for category, count in pain_rows
    )
    build_quality_rows = "\n".join(
        f"| {BUILD_QUALITY_LABELS_CN.get(category, category)} | {count} |"
        for category, count in build_quality_counts.most_common(5)
    )
    terms_md = ", ".join(f"`{word}`({count})" for word, count in terms[:12])

    return f"""# 亚马逊评论 + 商品运营分析报告

## 数据说明

- 商品表：`products_clean.csv`，有效商品 **{len(products):,}** 条
- 评论表：`reviews_long.csv`，长表评论 **{len(reviews):,}** 条
- 数据站点：Amazon.in 印度站，价格单位为 ₹ 卢比
- 平均商品评分：**{products['rating'].mean():.2f}**
- 折后价中位数：**₹{products['discounted_price'].median():,.0f}**
- 平均折扣：**{products['discount_percentage'].mean():.1f}%**

## 商品运营洞察

### 1. 品类供给集中度

| 品类 | 商品数 |
|---|---|
{count_rows}

### 2. 需求热度

| 品类 | 平均评论数 | 商品数 |
|---|---|---|
{demand_rows}

### 3. 价格带

折后价中位数为 ₹{products['discounted_price'].median():,.0f}，整体偏向低价高频电子配件。

### 4. 口碑结构

| 品类 | 平均评分 | 商品数 |
|---|---|---|
{rating_rows}

### 5. 综合选品候选

同时考虑商品数、平均评论数和平均评分，优先候选为：

| 品类 | 平均评论数 | 平均评分 | 商品数 |
|---|---|---|---|
{opportunity_rows}

## 评论痛点挖掘

> 说明：本结果基于规则词典的粗粒度情感分类，不代表绝对情感占比。负面痛点中的关键词可能存在口语化和上下文误判，建议对重点类目进行人工抽检。

- 正向：**{sentiment_counts.get('positive', 0):,}**
- 中性：**{sentiment_counts.get('neutral', 0):,}**
- 负向：**{sentiment_counts.get('negative', 0):,}**

| 痛点 | 提及次数 |
|---|---|
{pain_rows_md}

其中 `Build Quality` 可进一步细分为：

| 细分标签 | 提及次数 |
|---|---|
{build_quality_rows}

高频词：{terms_md}

## 运营结论

综合看，优先进入 **`{opportunity_rank.index[0]}`**：它在商品数、需求热度和口碑三个维度上更均衡。选品时兼顾 `{demand_rank.index[0]}` 的流量空间和 `{rating_rank.index[0]}` 的口碑优势，主推折后价 **₹{products['discounted_price'].median():,.0f}** 附近的产品，并优先解决 **{PAIN_LABELS_CN.get(pain_rows[0][0], pain_rows[0][0])}** 问题。
"""


def main() -> None:
    products, reviews = load_data()
    reviews = add_sentiment(reviews)
    make_charts(products, reviews)
    print(f"products={len(products):,}, reviews={len(reviews):,}")


if __name__ == "__main__":
    main()
