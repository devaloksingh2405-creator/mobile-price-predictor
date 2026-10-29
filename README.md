# Mobile Price Prediction System

Streamlit web app for the SRM IST ML project. It predicts a smartphone's price from
7 specifications using the project's final model (Gradient Boosting, 200 trees,
learning rate 0.1, max depth 3). If the phone has no user rating yet, it switches to
the specification-only model.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The web app (trains the model on startup, takes inputs, shows the price) |
| `requirements.txt` | Python packages the server installs |
| `mobile_price_cleaned.csv` | The cleaned dataset (807 rows, 543 after removing duplicates) |

## Deploy in about 5 minutes (Streamlit Community Cloud, free)

1. Make sure `mobile_price_cleaned.csv` is in this folder next to `app.py` (already included).
2. Go to github.com, sign in, click **New repository**, name it `mobile-price-predictor`,
   choose **Public**, click **Create repository**.
3. On the new repo page click **uploading an existing file**, drag in `app.py`,
   `requirements.txt` and `mobile_price_cleaned.csv`, then click **Commit changes**.
4. Go to share.streamlit.io, sign in with GitHub, click **Create app**, then
   **Yup, I have an app**.
5. Select repository `mobile-price-predictor`, branch `main`, main file `app.py`.
   Optionally set a custom App URL. Click **Deploy**.
6. Wait a few minutes. Your app goes live at `https://<name>.streamlit.app`.

To update the app later, edit or re-upload a file on GitHub; the app redeploys itself.

## Run on your own computer (optional)

```
pip install -r requirements.txt
streamlit run app.py
```

## Settings you can change in app.py

- `CURRENCY = "₹"`: set to `"₹"` if your prices are in rupees.
- `TEAM = "..."`: names shown in the About section.
