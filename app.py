@st.cache_data
def fetch_historical_data(instagram_account_id: str, token: str) -> pd.DataFrame:
    """
    Fetches historical media data (posts and metrics) from the Meta Graph API.
    Uses v19.0 and requests metrics so engagement is not constant.
    """
    BASE_URL = f"https://graph.facebook.com/v19.0/{instagram_account_id}/media"

    # NOW we request caption + like_count + comments_count
    fields = [
        "id",
        "caption",
        "timestamp",
        "media_type",
        "like_count",
        "comments_count",
    ]

    params = {
        "fields": ",".join(fields),
        "access_token": token,
        "limit": 100000,   # keep your original high limit
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API Request Error (media metrics): {e}")
        return pd.DataFrame()

    post_list = []

    for post in data.get("data", []):
        # real metrics from API
        likes = post.get("like_count")
        comments = post.get("comments_count")

        # if metrics are missing, skip to avoid constant fallbacks
        if likes is None or comments is None:
            continue

        engagement = likes + comments

        media_type = post.get("media_type", "IMAGE").lower()
        if media_type == "carousel_album":
            media_type = "carousel"

        try:
            post_time = datetime.fromisoformat(
                post["timestamp"].replace("+0000", "+00:00")
            )
        except Exception:
            continue

        caption = post.get("caption", "") or ""

        post_list.append(
            {
                "post_type": media_type,
                "weekday": post_time.strftime("%A"),
                "hour_utc": post_time.hour,
                "hashtags": caption.count("#"),
                "caption_length": len(caption),
                "likes": likes,
                "comments": comments,
                "shares": 0,
                "engagement": engagement,
                # this will be overwritten later with current_followers
                "engagement_rate": engagement / 100000,
                "id": post["id"],
                "timestamp": post_time,
                "caption": caption,
            }
        )

    return pd.DataFrame(post_list)
