import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

# .env лежить в корені проєкту (AUTORIA Project), а скрипт запускається з src/parser -
# тому явно вказуємо шлях, щоб load_dotenv() точно його знайшов незалежно від того,
# звідки саме запущено файл.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

# Ключ беремо з .env, а НЕ прописуємо в коді напряму (щоб не засвітити його знову).
# У файлі .env (в корені проєкту) має бути рядок: MY_API_KEY=твій_ключ
API_KEY = os.getenv("MY_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Не знайдено MY_API_KEY. Перевір, що в .env є рядок MY_API_KEY=... "
        "і що .env підвантажується (наприклад через python-dotenv)."
    )

FILE_NAME = os.path.join(BASE_DIR, 'data', 'new_cars_dataset.csv')

# Тимчасово зменшено для безпечного тестування - у тебе пакет всього на 100 запитів.
MAX_REQUESTS = 5


def safe_get(dct, key, default=None):
    """Безпечно дістає значення з словника, навіть якщо dct = None."""
    if not isinstance(dct, dict):
        return default
    value = dct.get(key, default)
    return value if value is not None else default


def get_collected_ids():
    """Повертає множину вже завантажених ID з файлу."""
    if os.path.exists(FILE_NAME):
        return set(pd.read_csv(FILE_NAME)['ID'].astype(str))
    return set()


collected = get_collected_ids()
print(f"Вже зібрано: {len(collected)} авто.")

requests_made = 0
to_download = []
page = 0

print("Шукаємо нові авто...")

# Збираємо ID, гортаючи сторінки пошуку
while requests_made < MAX_REQUESTS:
    search_url = f'https://developers.ria.com/auto/search?api_key={API_KEY}&category_id=1&countpage=100&page={page}'

    try:
        search_res = requests.get(search_url)
        requests_made += 1
    except Exception as e:
        print(f"Помилка під час пошуку: {e}")
        break

    if search_res.status_code == 200:
        result_data = search_res.json().get('result', {}).get('search_result', {})
        all_ids = result_data.get('ids', [])

        if not all_ids:
            print("Більше немає сторінок для пошуку (досягли кінця результатів).")
            break

        # Відбираємо тільки нові ID, яких ще немає у файлі
        new_ids = [str(i) for i in all_ids if str(i) not in collected]

        # Додаємо нові унікальні ID до загальної черги завантаження
        for car_id in new_ids:
            if car_id not in to_download:
                to_download.append(car_id)

        # Скільки ще запитів ми можемо зробити для отримання деталей авто?
        remaining_requests = MAX_REQUESTS - requests_made

        # Якщо ми зібрали достатньо ID, обрізаємо список до потрібної кількості і виходимо
        if len(to_download) >= remaining_requests:
            to_download = to_download[:remaining_requests]
            break

        page += 1
        time.sleep(1.5)  # збільшено з 0.5 - щоб уникнути rate limit по частоті запитів
    else:
        # ГОЛОВНА ЗМІНА: друкуємо тіло відповіді, щоб побачити реальну причину помилки
        print(f"Пошук не вдався. Код: {search_res.status_code}")
        print(f"Тіло відповіді: {search_res.text[:1000]}")

        if search_res.status_code == 429:
            print("429 Too Many Requests - або вичерпано ліміт пакету, "
                  "або перевищено дозволену частоту запитів (rps/rpm).")
        break

# Завантажуємо деталі для знайдених ID
if not to_download:
    print("Нових авто не знайдено або всі доступні вже в базі.")
else:
    print(f"Знайдено {len(to_download)} нових автомобілів. Починаємо завантаження деталей...")

    for car_id in to_download:
        if requests_made >= MAX_REQUESTS:
            break

        info_url = f'https://developers.ria.com/auto/info?api_key={API_KEY}&auto_id={car_id}'

        try:
            res = requests.get(info_url)
            requests_made += 1
        except Exception as e:
            print(f"Помилка запиту для ID {car_id}: {e}")
            time.sleep(3)
            continue

        if res.status_code == 200:
            data = res.json()
            auto_data = data.get('autoData', {}) or {}
            photo_data = data.get('photoData', {}) or {}
            level_data = data.get('levelData', {}) or {}
            phone_data = data.get('userPhoneData', {}) or {}
            description = data.get('description', '') or ''

            car_entry = {
                # --- Ідентифікатори ---
                'ID': car_id,
                'Mark': safe_get(data, 'markName', ''),
                'MarkId': safe_get(data, 'markId', 0),
                'Model': safe_get(data, 'modelName', ''),
                'ModelId': safe_get(data, 'modelId', 0),
                'CategoryId': safe_get(auto_data, 'categoryId', safe_get(data, 'categoryId', 0)),

                # --- Ціна ---
                'Price_USD': safe_get(data, 'USD', 0),
                'Price_UAH': safe_get(data, 'UAH', 0),
                'Price_EUR': safe_get(data, 'EUR', 0),
                'Main_Currency': safe_get(auto_data, 'mainCurrency', safe_get(data, 'mainCurrency', '')),
                'Auction_Possible': safe_get(data, 'auctionPossible', False),

                # --- Технічні характеристики ---
                'Year': safe_get(auto_data, 'year', safe_get(data, 'year', 0)),
                'Mileage': safe_get(auto_data, 'raceInt', 0),
                'Engine': safe_get(auto_data, 'engineVolume', 0),
                'Fuel': safe_get(auto_data, 'fuelName', ''),
                'Gearbox': safe_get(auto_data, 'gearboxName', ''),
                'Body': safe_get(auto_data, 'bodyName', ''),
                'Color': safe_get(auto_data, 'colorName', ''),
                'Drive': safe_get(auto_data, 'driveName', ''),
                'Wheel': safe_get(auto_data, 'wheelName', ''),
                'SeatsNumber': safe_get(auto_data, 'seatsNumber', 0),
                'DoorsNumber': safe_get(auto_data, 'doorsNumber', 0),
                'Custom': safe_get(auto_data, 'custom', safe_get(data, 'custom', 0)),
                'ConditionId': safe_get(auto_data, 'conditionId', 0),
                'StatusId': safe_get(auto_data, 'statusId', 0),
                'WithVideo': safe_get(auto_data, 'withVideo', safe_get(data, 'withVideo', False)),

                # --- Локація / продавець ---
                'City': safe_get(data, 'locationCityName', ''),
                'UserId': safe_get(data, 'userId', 0),
                'Is_Dealer': bool(safe_get(data, 'isAutoAddedByPartner', False)),
                'PartnerId': safe_get(data, 'partnerId', 0),
                'Phone_Verified': bool(safe_get(phone_data, 'phoneId', 0)),

                # --- Обмін / торг ---
                'Exchange_Possible': safe_get(data, 'exchangePossible', False),
                'Exchange_Type': safe_get(data, 'exchangeType', ''),

                # --- Оголошення / активність ---
                'Add_Date': safe_get(data, 'addDate', ''),
                'Update_Date': safe_get(data, 'updateDate', ''),
                'Expire_Date': safe_get(data, 'expireDate', ''),
                'Is_Sold': safe_get(auto_data, 'isSold', safe_get(data, 'isSold', False)),
                'From_Archive': safe_get(auto_data, 'fromArchive', safe_get(data, 'fromArchive', False)),
                'On_Moderation': safe_get(data, 'onModeration', False),
                'Chips_Count': safe_get(data, 'chipsCount', 0),  # лічильник (напр. перегляди/ліди), значення умовне

                # --- Топ / реклама ---
                'Top_Level': safe_get(level_data, 'level', 0),
                'Top_Label': safe_get(level_data, 'label', 0),
                'Hot_Type': safe_get(level_data, 'hotType', ''),

                # --- Фото та опис ---
                'Photos_Count': safe_get(photo_data, 'count', 0),
                'Description_Length': len(description),
            }

            os.makedirs(os.path.dirname(FILE_NAME), exist_ok=True)

            df = pd.DataFrame([car_entry])
            df.to_csv(FILE_NAME, mode='a', index=False, header=not os.path.exists(FILE_NAME))
            print(f"[{requests_made}/{MAX_REQUESTS}] Збережено: {car_entry['Mark']} {car_entry['Model']} (ID: {car_id})")

            time.sleep(1)
        else:
            print(f"Помилка на ID {car_id}: {res.status_code}")
            print(f"Тіло відповіді: {res.text[:500]}")

            if res.status_code == 429:
                print("Перевищено ліміт запитів API (429 Too Many Requests).")
                break
            time.sleep(2)

    print("\n--- ГОТОВО ---")
    print(f"Використано запитів: {requests_made} із {MAX_REQUESTS}.")