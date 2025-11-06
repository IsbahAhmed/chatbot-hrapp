# app/seed_docs.py
# Run this once to seed your HR policy docs into Chroma DB
from retriever import Retriever


SAMPLE_DOCS = [
{"id": "leave_policy", "text": "Annual leave: Employees get 20 paid days per year. Leave should be requested 7 days in advance except emergencies."},
{"id": "overtime", "text": "Overtime: Non-exempt employees will be compensated at 1.5x for hours worked beyond 40 hours/week. Prior approval is required."},
{"id": "compensation", "text": "Compensation reviews are annual and based on performance. Bonuses are discretionary."},
]


if __name__ == '__main__':
    r = Retriever()
    r.index_documents(SAMPLE_DOCS)
    print('Seeded docs')