# ACLED Integration Guide

## Overview

FamineSight integrates with the ACLED (Armed Conflict Location and Event Data) API to obtain conflict data for Somalia. This document details the authentication process and integration specifics.

## Authentication Process

### OAuth 2.0 Flow

FamineSight uses the OAuth 2.0 authentication method required by ACLED:

1. **Token Retrieval**: 
   - POST to `https://acleddata.com/oauth/token`
   - Parameters required:
     - `username`: Your ACLED email
     - `password`: Your ACLED password  
     - `grant_type`: "password"
     - `client_id`: "acled"
     - `scope`: "authenticated"

2. **Token Usage**:
   - Include token in Authorization header: `Bearer <token>`
   - Token valid for 24 hours
   - Automatic refresh on 401 errors

## Environment Configuration

### Required Variables

Add the following to your `.env` file:

```bash
ACLED_EMAIL=your_email@organization.org
ACLED_PASSWORD=your_secure_password
```

### Credential Security

- Credentials are stored in environment variables
- Never commit credentials to version control
- Use `.env` file with appropriate permissions (600)

## API Endpoints

### Main Data Endpoint
- **URL**: `https://acleddata.com/acled/read`
- **Method**: GET
- **Parameters**:
  - `country`: "Somalia"
  - `year`: "2010:2024" (historical range)
  - `limit`: 500 (maximum per page)
  - `page`: Pagination parameter

### Response Format
The API returns JSON with a `data` array containing event records with fields:
- `event_date`
- `event_type`
- `sub_event_type` 
- `actor1`
- `admin1` (region)
- `admin2` (district)
- `latitude`
- `longitude`
- `fatalities`
- `civilian_targeting`

## Data Processing

### Fetching Logic
1. **Authentication**: Get OAuth token
2. **Pagination**: Automatically handle multiple pages
3. **Rate Limiting**: Exponential backoff on 429 errors
4. **Error Handling**: Graceful fallback to synthetic data

### Data Cleaning
1. Parse `event_date` to datetime
2. Convert `fatalities` to integer
3. Extract `civilian_targeting` boolean
4. Filter invalid records
5. Aggregate to district-month level

## Error Handling

### Common Error Codes
- **401 Unauthorized**: Invalid credentials or token expired
- **429 Too Many Requests**: Rate limiting, implements exponential backoff
- **5xx Server Errors**: Temporary service issues, implements retry logic

### Fallback Mechanism
When API access fails:
1. System automatically falls back to synthetic data
2. All existing functionality continues
3. No interruption to system operation

## Rate Limiting

### Implementation
- **Exponential Backoff**: 2^page seconds between retries
- **Maximum Retries**: 5 attempts
- **Automatic Token Refresh**: On 401 errors
- **Graceful Degradation**: Continue with cached data when possible

## Testing Authentication

### Test Script
```python
# test_acled_auth.py
from src.data.acled_fetcher import get_acled_token

def test_authentication():
    token = get_acled_token()
    if token:
        print("✅ ACLED authentication successful")
        print(f"Token length: {len(token)}")
        return True
    else:
        print("❌ ACLED authentication failed")
        return False

if __name__ == "__main__":
    test_authentication()
```

## Security Considerations

### Credential Protection
- Store credentials in `.env` file
- Set proper file permissions (600)
- Never log credentials
- Use environment variable expansion only

### Token Security
- Tokens are cached in memory
- No persistent storage of tokens
- Automatic refresh on expiration
- Secure handling of authentication flow

### Data Security
- All processing done locally
- No external data transmission
- Secure API communication (HTTPS)
- Proper error handling to prevent credential leaks

## Troubleshooting

### Common Issues

1. **Authentication Failure**
   - Verify email and password
   - Check ACLED account status
   - Ensure account has API access

2. **Rate Limiting**
   - System implements automatic backoff
   - No manual intervention needed
   - Continue with synthetic data if rate limited

3. **Network Issues**
   - Verify internet connectivity
   - Check firewall settings
   - Ensure proper DNS resolution

### Debugging Tips

1. **Check Environment Variables**
```bash
grep ACLED .env
```

2. **Test Token Retrieval**
```bash
python -c "from src.data.acled_fetcher import get_acled_token; print(get_acled_token())"
```

3. **Verify API Access**
```bash
curl -G \
  --data-urlencode "username=your_email" \
  --data-urlencode "password=your_password" \
  --data-urlencode "grant_type=password" \
  --data-urlencode "client_id=acled" \
  --data-urlencode "scope=authenticated" \
  https://acleddata.com/oauth/token
```

## Integration Best Practices

### Production Use
1. **Secure Credential Storage**: Use environment variables only
2. **Error Monitoring**: Log authentication failures
3. **Fallback Testing**: Regularly test synthetic data fallback
4. **Token Management**: Monitor token expiration

### Development
1. **Use Synthetic Data**: Default to synthetic for development
2. **API Testing**: Test authentication separately
3. **Rate Limiting**: Understand API limits
4. **Data Validation**: Verify data quality after fetching

## Version Compatibility

### API Version
- Uses ACLED's current REST API
- OAuth 2.0 authentication
- JSON response format
- Standard HTTP status codes

### Client Version
- Python requests library for API calls
- Automatic retry logic
- Error handling and logging
- Token caching for efficiency