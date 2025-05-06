# Словарь данных - Telco Customer Churn

## Общая информация

- Количество строк: 7043
- Количество столбцов: 22
- Целевая переменная: Churn (отток клиентов)

## Описание признаков

| Признак | Тип данных | Описание | Уникальные значения | Пропущенные значения |
|---------|------------|----------|---------------------|----------------------|
| customerID | object | Уникальный идентификатор клиента | 7043 (7590-VHVEG, 5575-GNVDE, 3668-QPYBK, 7795-CFOCW, 9237-HQITU, ...) | 0 |
| gender | object | Пол клиента (Male/Female) | 2 (Female, Male) | 0 |
| SeniorCitizen | int64 | Является ли клиент пожилым (1) или нет (0) | 2 (0, 1) | 0 |
| Partner | object | Имеет ли клиент партнера (Yes/No) | 2 (Yes, No) | 0 |
| Dependents | object | Имеет ли клиент иждивенцев (Yes/No) | 2 (No, Yes) | 0 |
| tenure | int64 | Количество месяцев, в течение которых клиент пользуется услугами компании | 73 (1, 34, 2, 45, 8, ...) | 0 |
| PhoneService | object | Подключена ли телефонная услуга (Yes/No) | 2 (No, Yes) | 0 |
| MultipleLines | object | Подключены ли несколько телефонных линий (Yes/No/No phone service) | 3 (No phone service, No, Yes) | 0 |
| InternetService | object | Тип интернет-услуги (DSL, Fiber optic, No) | 3 (DSL, Fiber optic, No) | 0 |
| OnlineSecurity | object | Подключена ли услуга онлайн-безопасности (Yes/No/No internet service) | 3 (No, Yes, No internet service) | 0 |
| OnlineBackup | object | Подключена ли услуга онлайн-резервного копирования (Yes/No/No internet service) | 3 (Yes, No, No internet service) | 0 |
| DeviceProtection | object | Подключена ли услуга защиты устройства (Yes/No/No internet service) | 3 (No, Yes, No internet service) | 0 |
| TechSupport | object | Подключена ли услуга технической поддержки (Yes/No/No internet service) | 3 (No, Yes, No internet service) | 0 |
| StreamingTV | object | Подключена ли услуга потокового ТВ (Yes/No/No internet service) | 3 (No, Yes, No internet service) | 0 |
| StreamingMovies | object | Подключена ли услуга потокового видео (Yes/No/No internet service) | 3 (No, Yes, No internet service) | 0 |
| Contract | object | Тип контракта (Month-to-month, One year, Two year) | 3 (Month-to-month, One year, Two year) | 0 |
| PaperlessBilling | object | Использует ли клиент безбумажный расчет (Yes/No) | 2 (Yes, No) | 0 |
| PaymentMethod | object | Способ оплаты (Electronic check, Mailed check, Bank transfer, Credit card) | 4 (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) | 0 |
| MonthlyCharges | float64 | Ежемесячная плата | 1585 (29.85, 56.95, 53.85, 42.3, 70.7, ...) | 0 |
| TotalCharges | float64 | Общая сумма платежей | 6531 (29.85, 1889.5, 108.15, 1840.75, 151.65, ...) | 0 |
| Churn | object | Ушел ли клиент (Yes/No) | 2 (No, Yes) | 0 |
| Churn_Binary | int64 | Нет описания | 2 (0, 1) | 0 |
