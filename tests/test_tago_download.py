"""공식 API 응답 형태에 대한 합성 단위검사. 네트워크·실제 인증키 사용 없음."""
import json

import pytest

from collect_tago_routes import parse_response


def test_single_item_normalized_to_list():
    response = {"response": {"header": {"resultCode": "00"}, "body": {"items": {"item": {"routeid": "test_route"}}, "totalCount": 1}}}
    rows, total = parse_response(json.dumps(response))
    assert rows == [{"routeid": "test_route"}] and total == 1


def test_empty_items_is_not_fabricated_route():
    response = {"response": {"header": {"resultCode": "00"}, "body": {"items": "", "totalCount": 0}}}
    assert parse_response(json.dumps(response)) == ([], 0)


def test_gateway_error_never_treated_as_zero_routes():
    response = {"OpenAPI_ServiceResponse": {"cmmMsgHeader": {"returnReasonCode": "04"}}}
    with pytest.raises(ValueError, match="04"):
        parse_response(json.dumps(response))


def test_xml_error_never_treated_as_zero_routes():
    with pytest.raises(ValueError, match="30"):
        parse_response("<OpenAPI_ServiceResponse><cmmMsgHeader><returnReasonCode>30</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>")
