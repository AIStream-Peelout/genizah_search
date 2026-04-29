import requests
import json
import sys

# Try different possible local URLs if the domain is not accessible
POSSIBLE_URLS = ["http://localhost:8000", "http://127.0.0.1:8000", "https://api.cairogenizah.ai"]

def find_api_url():
    for url in POSSIBLE_URLS:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Found working API at {url}")
                return url
        except requests.RequestException:
            continue
    return None

def test_keyword_search_with_filters(api_url):
    print("\n--- Testing Keyword Search with Filters ---")
    payload = {
        "query": "marriage",
        "filters": {"language": "Judaeo-Arabic"},
        "num_results": 5
    }
    print(f"Request payload: {json.dumps(payload)}")
    try:
        response = requests.post(f"{api_url}/search-keyword", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Found {data['count']} results")
            print(f"Applied filters in response: {json.dumps(data['filters_applied'])}")
            
            # Check if results match the filter
            mismatches = 0
            for res in data['results']:
                lang = res['metadata'].get('language')
                if lang != 'Judaeo-Arabic':
                    mismatches += 1
                    print(f"❌ Mismatch: doc {res['doc_id']} has language {lang}")
            
            if mismatches == 0:
                print("✅ All results match requested filters")
            return True
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_shelfmark_search_with_filters(api_url):
    print("\n--- Testing Shelfmark Search with Filters ---")
    payload = {
        "shelf_mark": "T-S",
        "exact_match": False,
        "filters": {"collection": "Taylor-Schechter (T-S)"},
        "num_results": 5
    }
    try:
        response = requests.post(f"{api_url}/search-shelfmark", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Found {data['count']} results")
            print(f"Applied filters in response: {json.dumps(data['filters_applied'])}")
            
            # Check if results match the filter
            mismatches = 0
            for res in data['results']:
                coll = res['metadata'].get('collection')
                if coll != 'Taylor-Schechter (T-S)':
                    mismatches += 1
                    print(f"❌ Mismatch: doc {res['doc_id']} has collection {coll}")
            
            if mismatches == 0:
                print("✅ All results match requested filters")
            return True
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_semantic_search(api_url):
    print("\n--- Testing Semantic Search ---")
    payload = {
        "query": "marriage contract",
        "num_results": 5
    }
    print(f"Request payload: {json.dumps(payload)}")
    try:
        response = requests.post(f"{api_url}/search", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: Found {data['count']} results")
            print(f"Applied filters in response: {json.dumps(data.get('filters_applied'))}")
            if data['results']:
                print(f"First result: {data['results'][0]['doc_id']} (Score: {data['results'][0]['similarity_score']})")
            return True
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    api_url = find_api_url()
    if not api_url:
        print("❌ Could not find a working API endpoint. Please ensure the backend is running.")
        sys.exit(1)
    
    success_kw = test_keyword_search_with_filters(api_url)
    success_sm = test_shelfmark_search_with_filters(api_url)
    success_sem = test_semantic_search(api_url)
    
    if success_kw and success_sm and success_sem:
        print("\n🎉 All backend filter tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed.")
        sys.exit(1)
