import requests
import pandas as pd
import os
import time
import json
import logging
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Шляхи та конфігурація
# ----------------------------------------------------------------------------

# .env лежить в корені проєкту (Auto-Ria-project), а скрипт запускається з
# src/parser - тому явно вказуємо шлях, щоб load_dotenv() точно його знайшов
# незалежно від того, звідки саме запущено файл.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

API_KEY = os.getenv("MY_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Не знайдено MY_API_KEY. Перевір, що в .env є рядок MY_API_KEY=... "
        "і що .env підвантажується (наприклад через python-dotenv)."
    )

FILE_NAME = os.path.join(BASE_DIR, 'data', 'new_cars_dataset_2.csv')
PHOTOS_DIR = os.path.join(BASE_DIR, 'data', 'photos')
STATE_FILE = os.path.join(BASE_DIR, 'data', 'parser_state.json')
LOG_FILE = os.path.join(BASE_DIR, 'data', 'parser.log')

# Пакет: 100 000 запитів/місяць, 5 000 запитів/годину.
# Пагінація пошуку рахується в той самий бюджет, що й /auto/info - тому ціль
# трохи менша за рівний місячний ліміт, а не 100000 впритул.
MAX_REQUESTS = 98000

REQUEST_TIMEOUT = 10
SITE_BASE_URL = "https://auto.ria.com"

# Скільки фото качати паралельно на етапі "довантаження фото". Це окремі
# запити на CDN (не на API), тому паралелізм тут безпечний.
PHOTO_DOWNLOAD_WORKERS = 8

session = requests.Session()

# ----------------------------------------------------------------------------
# Логування - і в консоль, і у файл (процес буде йти багато годин у фоні,
# тож консоль не завжди буде під рукою).
# ----------------------------------------------------------------------------

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("autoria_parser")


def safe_get(dct, key, default=None):
    """Безпечно дістає значення з словника, навіть якщо dct = None."""
    if not isinstance(dct, dict):
        return default
    value = dct.get(key, default)
    return value if value is not None else default


def fetch(url):
    """
    Виконує GET-запит до API із тайм-аутом і обробкою мережевих помилок / 429.

    Повертає requests.Response при успіху, або None, якщо запит не вдався
    (мережева помилка) чи перевищено ліміт запитів (429). У випадку 429
    чекає час із заголовка Retry-After (або 10с за замовчуванням) і повертає
    None - свідомо НЕ повторює запит автоматично, щоб не витрачати бюджет.
    """
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Помилка мережі: {e}")
        return None

    if response.status_code == 429:
        wait_seconds = int(response.headers.get("Retry-After", 10))
        logger.warning(f"429 Too Many Requests - чекаємо {wait_seconds} с.")
        time.sleep(wait_seconds)
        return None

    return response


# ----------------------------------------------------------------------------
# Стан пагінації - щоб багатогодинний збір можна було продовжити після
# перезапуску, а не сканувати сторінки пошуку з нуля щоразу.
# ----------------------------------------------------------------------------

def load_state():
    """Повертає останню опрацьовану сторінку пошуку (0, якщо стану ще немає)."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f).get('last_page', 0)
    return 0


def save_state(last_page):
    """Зберігає номер сторінки пошуку, з якої продовжити наступного разу."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump({'last_page': last_page}, f)


def get_collected_ids():
    """Повертає множину вже завантажених ID з файлу."""
    if os.path.exists(FILE_NAME):
        return set(pd.read_csv(FILE_NAME)['ID'].astype(str))
    return set()


# ----------------------------------------------------------------------------
# Основний цикл: пошук + деталі, сторінка за сторінкою
# ----------------------------------------------------------------------------

def run_collection(budget):
    """
    Пошук нових ID і завантаження деталей йдуть в одному циклі, сторінка за
    сторінкою: знайшли ID на сторінці -> одразу довантажили деталі -> лог
    прогресу -> наступна сторінка. Це навмисно, а не двофазно (спершу весь
    пошук, потім усі деталі) з двох причин:

    1. Видимий прогрес. При великому budget "спершу весь пошук" міг би йти
       сотні сторінок без жодного логу, поки не набере достатньо ID - і
       процес виглядав би "зависшим", хоча просто мовчки працював.
    2. Стійкість до переривання. Деталі зберігаються в CSV одразу, а не
       після завершення всього пошуку - тож переривання (Ctrl+C, крах)
       ніколи не залишає бюджет витраченим "в нікуди", без збережених даних.

    Повертає requests_made.
    """
    collected = get_collected_ids()
    logger.info(f"Вже зібрано: {len(collected)} авто.")

    requests_made = 0
    page = load_state()
    current_year = datetime.datetime.now().year
    total_saved = 0

    logger.info(f"Починаємо збір, стартова сторінка пошуку: {page}...")

    while requests_made < budget:
        search_url = (
            f'https://developers.ria.com/auto/search'
            f'?api_key={API_KEY}&category_id=1&countpage=100&page={page}'
        )

        requests_made += 1
        search_res = fetch(search_url)

        if search_res is None:
            break

        if search_res.status_code != 200:
            logger.error(f"Пошук не вдався. Код: {search_res.status_code}")
            logger.error(f"Тіло відповіді: {search_res.text[:1000]}")
            break

        result_data = search_res.json().get('result', {}).get('search_result', {})
        all_ids = result_data.get('ids', [])

        if not all_ids:
            logger.info("Більше немає сторінок для пошуку (досягли кінця результатів).")
            break

        new_ids = [str(i) for i in all_ids if str(i) not in collected]

        for car_id in new_ids:
            if requests_made >= budget:
                break

            info_url = f'https://developers.ria.com/auto/info?api_key={API_KEY}&auto_id={car_id}'

            requests_made += 1
            res = fetch(info_url)

            if res is None:
                time.sleep(2)
                continue

            if res.status_code == 200:
                car_entry = parse_car_entry(res.json(), car_id, current_year)
                save_entry(car_entry)
                # Одразу позначаємо як зібраний - інакше той самий ID міг би
                # знову потрапити в new_ids на іншій сторінці (сортування
                # видачі може зсуватись під час довгого прогону) і завантажитись
                # повторно, витрачаючи бюджет на дублікат.
                collected.add(car_id)
                total_saved += 1
                time.sleep(1)
            else:
                logger.error(f"Помилка на ID {car_id}: {res.status_code}")
                logger.error(f"Тіло відповіді: {res.text[:500]}")
                time.sleep(2)

        page += 1
        save_state(page)
        logger.info(
            f"[Сторінка {page}] Всього збережено нових авто: {total_saved}. "
            f"Використано запитів: {requests_made}/{budget}.")
        time.sleep(1.5)

    logger.info(f"--- Збір завершено. Нових авто збережено: {total_saved}. ---")
    return requests_made


# ----------------------------------------------------------------------------
# Довантаження фото - окремо від API, паралельно, без витрати бюджету
# ----------------------------------------------------------------------------

def parse_car_entry(data, car_id, current_year):
    """Перетворює сирий JSON з /auto/info у плаский словник для CSV."""
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

    car_year = int(safe_get(auto_data, 'year', safe_get(data, 'year', current_year)))
    age = current_year - car_year if car_year > 1900 else 0
    mileage_k = safe_get(auto_data, 'raceInt', 0)
    link_to_view = safe_get(data, 'linkToView', '')
    main_photo_url = safe_get(photo_data, 'seoLinkB', safe_get(photo_data, 'seoLinkF', ''))

    return {
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
        # Саме фото качається окремо, після основного циклу (див.
        # download_pending_photos нижче) - тут лише плейсхолдер.
        'Main_Photo_Path': '',
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
            word in desc_lower for word in ['ідеальний', 'идеальное', 'сів і поїхав', 'не фарбован']),
    }


def save_entry(car_entry):
    """Дописує один рядок у CSV (mode='a'), створюючи заголовок при першому запуску."""
    os.makedirs(os.path.dirname(FILE_NAME), exist_ok=True)
    df = pd.DataFrame([car_entry])
    df.to_csv(
        FILE_NAME,
        mode='a',
        index=False,
        header=not os.path.exists(FILE_NAME),
        encoding='utf-8-sig',
    )


# ----------------------------------------------------------------------------
# Фаза 3: довантаження фото - окремо від API, паралельно, без витрати бюджету
# ----------------------------------------------------------------------------

def _download_one_photo(car_id, url):
    """Качає одне фото. Повертає (car_id, relative_path) або (car_id, '') при невдачі."""
    try:
        photo_res = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Не вдалося завантажити фото для ID {car_id}: {e}")
        return car_id, ''

    if photo_res.status_code != 200:
        logger.warning(f"Фото для ID {car_id} недоступне (код {photo_res.status_code}).")
        return car_id, ''

    os.makedirs(PHOTOS_DIR, exist_ok=True)
    extension = os.path.splitext(url)[1].split('?')[0] or '.jpg'
    photo_path = os.path.join(PHOTOS_DIR, f"{car_id}{extension}")

    with open(photo_path, 'wb') as f:
        f.write(photo_res.content)

    return car_id, os.path.relpath(photo_path, BASE_DIR)


def download_pending_photos(max_workers=PHOTO_DOWNLOAD_WORKERS):
    """
    Довантажує головні фото для всіх рядків, де Main_Photo_URL є, а
    Main_Photo_Path ще порожній. Це окремі запити на CDN, а не на API -
    тому виконуються паралельно і не витрачають ліміт MAX_REQUESTS.
    """
    if not os.path.exists(FILE_NAME):
        return

    df = pd.read_csv(FILE_NAME, dtype={'ID': str})
    pending = df[(df['Main_Photo_Path'].fillna('') == '') & (df['Main_Photo_URL'].fillna('') != '')]

    if pending.empty:
        logger.info("Усі фото вже завантажені.")
        return

    logger.info(f"Довантажуємо {len(pending)} фото...")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one_photo, row['ID'], row['Main_Photo_URL']): row['ID']
            for _, row in pending.iterrows()
        }
        for future in as_completed(futures):
            car_id, path = future.result()
            results[car_id] = path

    df['Main_Photo_Path'] = df.apply(
        lambda row: results.get(row['ID'], row['Main_Photo_Path']), axis=1
    )
    df.to_csv(FILE_NAME, index=False, encoding='utf-8-sig')

    saved_count = sum(1 for path in results.values() if path)
    logger.info(f"Фото завантажено: {saved_count} з {len(pending)}.")


# ----------------------------------------------------------------------------
# Точка входу
# ----------------------------------------------------------------------------

def main():
    requests_made = run_collection(MAX_REQUESTS)

    logger.info(f"Використано запитів: {requests_made} із {MAX_REQUESTS}.")

    download_pending_photos()

    logger.info("--- ГОТОВО ---")


if __name__ == "__main__":
    main()