r"""
从原始亚马逊 CSV 清洗并拆分为商品表和评论长表。

清洗口径：
- 商品表保留 11 个商品字段，去掉混合在商品行里的评论字段。
- 评论表以非空 review_title 的数量作为评论条数。
- review_content 不能简单按逗号拆分，因为正文中也可能有逗号。
- 当前策略：按逗号切块，去掉空块，再按评论条数截断或补 NaN。
"""

from pathlib import Path

import pandas as pd


RAW_INPUT = Path(r"D:\项目\amazon.csv")


PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "about_product",
    "img_link",
    "product_link",
]


def split_list(value) -> list[str]:
    """按逗号拆成列表，并去掉首尾空格。"""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",")]


def split_review_content(value: str, expected_count: int) -> list[str | None]:
    """拆分评论正文，并尽可能对齐到 expected_count 条。"""
    chunks = [part for part in split_list(value) if part]
    if len(chunks) >= expected_count:
        return chunks[:expected_count]
    return chunks + [None] * (expected_count - len(chunks))


def build_reviews_long(raw: pd.DataFrame) -> pd.DataFrame:
    """把一行多评论的原始商品数据，转成一行一条评论的长表。"""
    rows: list[dict] = []

    for _, product in raw.iterrows():
        review_ids = split_list(product["review_id"])
        user_ids = split_list(product["user_id"])
        user_names = split_list(product["user_name"])
        review_titles = [
            title for title in split_list(product["review_title"]) if title
        ]
        review_contents = split_review_content(
            product["review_content"], len(review_titles)
        )

        for index, review_title in enumerate(review_titles):
            rows.append(
                {
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "rating": product["rating"],
                    "user_id": user_ids[index] if index < len(user_ids) else None,
                    "user_name": user_names[index] if index < len(user_names) else None,
                    "review_id": review_ids[index] if index < len(review_ids) else None,
                    "review_title": review_title,
                    "review_content": review_contents[index],
                }
            )

    return pd.DataFrame(rows)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """从原始 CSV 在内存中生成商品表和评论长表。"""
    raw = pd.read_csv(RAW_INPUT, encoding="utf-8")

    products = raw[PRODUCT_COLUMNS].copy()
    reviews = build_reviews_long(raw)
    return products, reviews


if __name__ == "__main__":
    products, reviews = load_raw_data()
    print(f"Prepared products: {len(products):,}")
    print(f"Prepared reviews:  {len(reviews):,}")
