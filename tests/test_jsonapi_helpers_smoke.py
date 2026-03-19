import pytest

from colette.jsonapi import (
    colette_bad_request_400,
    colette_created_201,
    colette_forbidden_403,
    colette_internal_error_500,
    colette_job_not_found_1003,
    colette_no_data_1001,
    colette_not_found_404,
    colette_ok_200,
    colette_service_bad_request_1006,
    colette_service_input_bad_request_1004,
    colette_service_input_error_1005,
    colette_service_llmlib_error_1007,
    colette_service_not_found_1002,
    colette_unknown_library_1000,
    render_status,
)


@pytest.mark.smoke
def test_render_status_variants():
    ok = render_status(200, "OK")
    err = render_status(500, "Internal Error", colette_code=500, colette_message="boom")

    assert ok.code == 200
    assert ok.colette_code is None
    assert err.code == 500
    assert err.colette_code == 500
    assert err.colette_message == "boom"


@pytest.mark.smoke
def test_jsonapi_error_response_helpers():
    responses = [
        colette_ok_200(),
        colette_created_201(),
        colette_bad_request_400("bad"),
        colette_forbidden_403(),
        colette_not_found_404(),
        colette_internal_error_500("err"),
        colette_unknown_library_1000("hf"),
        colette_no_data_1001(),
        colette_service_not_found_1002("svc"),
        colette_job_not_found_1003(),
        colette_service_input_bad_request_1004(),
        colette_service_input_error_1005("x"),
        colette_service_bad_request_1006("y"),
        colette_service_llmlib_error_1007("z"),
    ]

    codes = [r.status.code for r in responses if r.status is not None]
    assert 200 in codes
    assert 201 in codes
    assert 400 in codes
    assert 403 in codes
    assert 404 in codes
    assert 500 in codes
