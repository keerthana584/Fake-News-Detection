
import pandas as pd

data = {
    "text": [
        "The government announced a new education policy today",
        "Scientists discovered a new planet in space",
        "Breaking news celebrity becomes president of the moon",
        "Doctors confirm drinking petrol cures all diseases",
        "The school announced examination dates for students",
        "A secret machine can make everyone rich overnight"
    ],
    "label": [
        "REAL",
        "REAL",
        "FAKE",
        "FAKE",
        "REAL",
        "FAKE"
    ]
}

df = pd.DataFrame(data)

df.to_csv("news_dataset.csv", index=False)

print("Dataset created successfully!")
print(df)

