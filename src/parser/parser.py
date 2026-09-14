import requests
import pandas as pd
import os
import time
import datetime  # Додано для розрахунку віку авто
from dotenv import load_dotenv

# .env лежить в корені проєкту (AUTORIA Project), а скрипт запускається з src/parser -
# тому явно вказуємо шлях, щоб load_dotenv() точно його знайшов незалежно від того,
# звідки саме запущено файл.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

# Ключ беремо з .env, а НЕ прописуємо в коді напряму (щоб не засвітити його знову).
API_KEY = os.getenv("MY_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Не знайдено MY_API_KEY. Перевір, що в .env є рядок MY_API_KEY=... "
        "і що .env підвантажується (наприклад через python-dotenv)."
    )

FILE_NAME = os.path.join(BASE_DIR, 'data', 'new_cars_dataset_2.csv')

# Сюди зберігаються головні фото авто (по одному файлу на ID).
PHOTOS_DIR = os.path.join(BASE_DIR, 'data', 'photos')

# Тимчасово зменшено для безпечного тестування - у тебе пакет всього на 100 запитів.
MAX_REQUESTS = 5

# Скільки секунд чекати відповіді сервера, перш ніж вважати запит невдалим.
REQUEST_TIMEOUT = 10

# Базовий URL сайту - потрібен, щоб зібрати повне посилання з linkToView.
SITE_BASE_URL = "https://auto.ria.com"

# Одна сесія на весь скрипт - трохи швидше за окремий requests.get() щоразу,
# бо з'єднання з сервером повторно використовується.
session = requests.Session()


def safe_get(dct, key, default=None):
    """Безпечно дістає значення з словника, навіть якщо dct = None."""
    if not isinstance(dct, dict):
        return default
    value = dct.get(key, default)
    return value if value is not None else default


def fetch(url):
    """
    Виконує GET-запит із тайм-аутом і обробкою мережевих помилок / 429.

    Повертає requests.Response при успіху, або None, якщо запит не вдався
    (мережева помилка) чи перевищено ліміт запитів (429). У випадку 429
    чекає час із заголовка Retry-After (або 10с за замовчуванням) і повертає
    None - свідомо НЕ повторює запит автоматично, щоб не витрачати залишок
    і без того малого ліміту пакету.
    """
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"Помилка мережі: {e}")
        return None

    if response.status_code == 429:
        wait_seconds = int(response.headers.get("Retry-After", 10))
        print(f"429 Too Many Requests - чекаємо {wait_seconds} с (ліміт пакету або частоти запитів).")
        time.sleep(wait_seconds)
        return None

    return response


def download_main_photo(url, car_id):
    """
    Завантажує головне фото авто і зберігає його в PHOTOS_DIR як <car_id>.<розширення>.

    Це окремий запит на CDN (cdn.riastatic.com), а НЕ на сам API - тому він
    не витрачає ліміт MAX_REQUESTS. Повертає шлях до збереженого файлу
    (відносно BASE_DIR) або '' , якщо фото немає чи завантажити не вдалося.
    """
    if not url:
        return ''

    try:
        photo_res = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"Не вдалося завантажити фото для ID {car_id}: {e}")
        return ''

    if photo_res.status_code != 200:
        print(f"Фото для ID {car_id} недоступне (код {photo_res.status_code}).")
        return ''

    os.makedirs(PHOTOS_DIR, exist_ok=True)

    extension = os.path.splitext(url)[1].split('?')[0] or '.jpg'
    photo_path = os.path.join(PHOTOS_DIR, f"{car_id}{extension}")

    with open(photo_path, 'wb') as f:
        f.write(photo_res.content)

    return os.path.relpath(photo_path, BASE_DIR)


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

    requests_made += 1
    search_res = fetch(search_url)

    if search_res is None:
        # Мережева помилка або 429 - fetch() вже все роздрукував/зачекав.
        break

    if search_res.status_code == 200:
        result_data = search_res.json().get('result', {}).get('search_result', {})
        all_ids = result_data.get('ids', [])

        if not all_ids:
            print("Більше немає сторінок для пошуку (досягли кінця результатів).")
            break

        # Відбираємо тільки нові ID, яких ще немає у файлі
        new_ids = [str(i) for i in all_ids if str(i) not in collected]

        for car_id in new_ids:
            if car_id not in to_download:
                to_download.append(car_id)

        remaining_requests = MAX_REQUESTS - requests_made

        if len(to_download) >= remaining_requests:
            to_download = to_download[:remaining_requests]
            break

        page += 1
        time.sleep(1.5)
    else:
        print(f"Пошук не вдався. Код: {search_res.status_code}")
        print(f"Тіло відповіді: {search_res.text[:1000]}")
        break

# Завантажуємо деталі для знайдених ID
if not to_download:
    print("Нових авто не знайдено або всі доступні вже в базі.")
else:
    print(f"Знайдено {len(to_download)} нових автомобілів. Починаємо завантаження деталей...")
    current_year = datetime.datetime.now().year

    for car_id in to_download:
        if requests_made >= MAX_REQUESTS:
            break

        info_url = f'https://developers.ria.com/auto/info?api_key={API_KEY}&auto_id={car_id}'

        requests_made += 1
        res = fetch(info_url)

        if res is None:
            # Мережева помилка або 429. При мережевій помилці варто спробувати
            # наступне авто; при 429 fetch() вже почекав Retry-After.
            time.sleep(2)
            continue

        if res.status_code == 200:
            data = res.json()

            # --- Парсинг основних блоків ---
            auto_data = data.get('autoData', {}) or {}
            photo_data = data.get('photoData', {}) or {}
            level_data = data.get('levelData', {}) or {}
            phone_data = data.get('userPhoneData', {}) or {}
            state_data = data.get('stateData', {}) or {}
            checked_vin_data = data.get('checkedVin', {}) or {}
            dealer_data = data.get('dealer', {}) or {}
            description = data.get('description', '') or ''
            desc_lower = description.lower()
            options_list = safe_get(data, 'optionIdList', [])
            badges_list = safe_get(data, 'badges', [])

            # --- Розрахунок похідних змінних ---
            car_year = int(safe_get(auto_data, 'year', safe_get(data, 'year', current_year)))
            age = current_year - car_year if car_year > 1900 else 0
            mileage_k = safe_get(auto_data, 'raceInt', 0)
            link_to_view = safe_get(data, 'linkToView', '')
            main_photo_url = safe_get(photo_data, 'seoLinkB', safe_get(photo_data, 'seoLinkF', ''))
            main_photo_path = download_main_photo(main_photo_url, car_id)

            car_entry = {
                # --- Ідентифікатори ---
                'ID': car_id,
                'Mark': safe_get(data, 'markName', ''),
                'MarkId': safe_get(data, 'markId', 0),
                'Model': safe_get(data, 'modelName', ''),
                'ModelId': safe_get(data, 'modelId', 0),
                'Modification': safe_get(auto_data, 'version', ''),
                'CategoryId': safe_get(auto_data, 'categoryId', safe_get(data, 'categoryId', 0)),

                # --- Ціна ---
                'Price_USD': safe_get(data, 'USD', 0),
                'Price_UAH': safe_get(data, 'UAH', 0),
                'Price_EUR': safe_get(data, 'EUR', 0),
                'Main_Currency': safe_get(auto_data, 'mainCurrency', safe_get(data, 'mainCurrency', '')),
                'Auction_Possible': safe_get(data, 'auctionPossible', False),

                # --- Технічні характеристики ---
                'Year': car_year,
                'Age': age,
                'Mileage_K': mileage_k,
                'Mileage_Per_Year': mileage_k / (age if age > 0 else 1),
                'Engine_Volume': safe_get(auto_data, 'engineVolume', 0.0),
                'Fuel_Name': safe_get(auto_data, 'fuelName', ''),
                'Fuel_Id': safe_get(auto_data, 'fuelId', 0),
                'Gearbox_Name': safe_get(auto_data, 'gearboxName', ''),
                'Gearbox_Id': safe_get(auto_data, 'gearboxId', 0),
                'Body_Name': safe_get(auto_data, 'bodyName', ''),
                'Body_Id': safe_get(auto_data, 'bodyId', 0),
                'Color_Name': safe_get(auto_data, 'colorName', ''),
                'Color_Id': safe_get(auto_data, 'colorId', 0),
                'Drive_Name': safe_get(auto_data, 'driveName', ''),
                'Drive_Id': safe_get(auto_data, 'driveId', 0),
                'Wheel_Name': safe_get(auto_data, 'wheelName', ''),
                'SeatsNumber': safe_get(auto_data, 'seatsNumber', 0),
                'DoorsNumber': safe_get(auto_data, 'doorsNumber', 0),

                # --- Походження, стан та перевірки ---
                'Custom': safe_get(auto_data, 'custom', safe_get(data, 'custom', 0)),
                'Country_Origin_Id': safe_get(data, 'countryId', 0),
                'ConditionId': safe_get(auto_data, 'conditionId', 0),
                'StatusId': safe_get(auto_data, 'statusId', 0),
                'Is_Crashed': safe_get(auto_data, 'isDtp', False) or safe_get(data, 'isCrashed', False),
                'Damage_Description': safe_get(data, 'damageDescription', ''),
                'Is_Leasing': bool(safe_get(data, 'isLeasing', 0)),
                'Has_VIN': safe_get(data, 'hasVIN', False),
                'Is_Checked_VIN': safe_get(data, 'isCheckedVin', False),
                'VIN': safe_get(data, 'VIN', ''),
                'VIN_Has_Restrictions': safe_get(checked_vin_data, 'hasRestrictions', False),
                'Has_Plate': safe_get(data, 'hasPlate', False),
                'Is_Checked_Plate': safe_get(data, 'isCheckedPlate', False),

                # --- Локація та продавець ---
                'State_Id': safe_get(state_data, 'stateId', 0),
                'State_Name': safe_get(state_data, 'regionName', ''),
                'City_Id': safe_get(state_data, 'cityId', 0),
                'City': safe_get(state_data, 'name', safe_get(data, 'locationCityName', '')),
                'UserId': safe_get(data, 'userId', 0),
                'Is_Dealer': bool(safe_get(data, 'isAutoAddedByPartner', False)),
                'PartnerId': safe_get(data, 'partnerId', 0),
                'Dealer_Name': safe_get(dealer_data, 'name', ''),
                'Seller_Type': safe_get(data, 'sellerType', ''),
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
                'Views_Total': safe_get(data, 'views', 0),
                'Views_Today': safe_get(data, 'viewsToday', 0),
                'Bookmarks_Count': safe_get(data, 'bookmarks', 0),
                'Chips_Count': safe_get(data, 'chipsCount', 0),

                # --- Топ / реклама ---
                'Top_Level': safe_get(level_data, 'level', 0),
                'Top_Label': safe_get(level_data, 'label', 0),
                'Hot_Type': safe_get(level_data, 'hotType', ''),
                'Badges_Count': len(badges_list) if isinstance(badges_list, list) else 0,

                # --- Контент, посилання та Опис ---
                'Ad_Link': f"{SITE_BASE_URL}{link_to_view}" if link_to_view else '',
                'Main_Photo_URL': main_photo_url,
                'Main_Photo_Path': main_photo_path,
                'Photos_Count': safe_get(photo_data, 'count', 0),
                'With_Video': safe_get(auto_data, 'withVideo', safe_get(data, 'withVideo', False)),
                'Description_Length': len(description),
                'Options_Count': len(options_list),

                # --- NLP Фічі на основі опису ---
                'Desc_Urgent': any(word in desc_lower for word in ['терміново', 'срочно']),
                'Desc_Bargain': 'торг' in desc_lower,
                'Desc_First_Owner': any(
                    word in desc_lower for word in ['перший власник', 'один власник', 'первый владелец', 'з салону']),
                'Desc_Ideal': any(
                    word in desc_lower for word in ['ідеальний', 'идеальное', 'сів і поїхав', 'не фарбован'])
            }

            os.makedirs(os.path.dirname(FILE_NAME), exist_ok=True)

            df = pd.DataFrame([car_entry])
            df.to_csv(
                FILE_NAME,
                mode='a',
                index=False,
                header=not os.path.exists(FILE_NAME),
                encoding='utf-8-sig',  # без цього кирилиця в описах "ламається" в Excel на Windows
            )
            print(
                f"[{requests_made}/{MAX_REQUESTS}] Збережено: {car_entry['Mark']} {car_entry['Model']} (ID: {car_id})")

            time.sleep(1)
        else:
            print(f"Помилка на ID {car_id}: {res.status_code}")
            print(f"Тіло відповіді: {res.text[:500]}")
            time.sleep(2)

    print("\n--- ГОТОВО ---")
    print(f"Використано запитів: {requests_made} із {MAX_REQUESTS}.")