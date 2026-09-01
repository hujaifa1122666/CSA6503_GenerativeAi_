# Install: pip install pandas
import pandas as pd

data = {
    "Student Name": ["Ali", "Rahul", "Priya", "John", "Sara"],
    "Grade": ["A", "B", "A+", "B+", "A"]
}

df = pd.DataFrame(data)

print("Student Grades:")
print(df)
