import asyncio
import os
import shutil
import traceback

from pydantic import DirectoryPath

from .apidata import APIData, APIResponse, InfoObj, StatusObj
from .apistrategy import APIStrategy

# model
from .inputconnector import (
    InputConnectorBadParamException,
    InputConnectorInternalException,
)
from .llmlib import LLMLibBadParamException, LLMLibInternalException
from .llmmodel import LLMModelBadParamException, LLMModelInternalException
from .llmservice import createLLMService
from .services import ServiceBadParamException


def render_status(code: int, status: str, colette_code: int = -1, colette_message: str = ""):
    if colette_code != -1:
        return StatusObj(
            code=code,
            status=status,
            colette_code=colette_code,
            colette_message=colette_message,
        )
    else:
        return StatusObj(code=code, status=status)


# Errors
def colette_ok_200() -> APIResponse:
    return APIResponse(status=render_status(code=200, status="OK"))


def colette_created_201() -> APIResponse:
    return APIResponse(status=render_status(code=201, status="Created"))


def colette_bad_request_400(message: str):
    return APIResponse(status=render_status(code=400, status="Bad Request", colette_code=400, colette_message=message))


def colette_forbidden_403() -> APIResponse:
    return APIResponse(status=render_status(code=403, status="Forbidden"))


def colette_not_found_404() -> APIResponse:
    return APIResponse(status=render_status(code=404, status="Not Found"))


def colette_internal_error_500(message: str) -> APIResponse:
    return APIResponse(
        status=render_status(code=500, status="Internal Error", colette_code=500, colette_message=message)
    )


# Specific errors
def colette_unknown_library_1000(what: str) -> APIResponse:
    return APIResponse(
        status=render_status(
            code=404,
            status="Not Found",
            colette_code=1000,
            colette_message="Unknown Library: " + what,
        )
    )


def colette_no_data_1001() -> APIResponse:
    return APIResponse(
        status=render_status(code=400, status="Bad Request", colette_code=1001, colette_message="No Data")
    )


def colette_service_not_found_1002(sname: str) -> APIResponse:
    return APIResponse(
        status=render_status(
            code=404,
            status="Not Found",
            colette_code=1002,
            colette_message="Service Not Found: " + sname,
        )
    )


def colette_job_not_found_1003() -> APIResponse:
    return APIResponse(
        status=render_status(
            code=404,
            status="Not Found",
            colette_code=1003,
            colette_message="Job Not Found",
        )
    )


def colette_service_input_bad_request_1004() -> APIResponse:
    return APIResponse(
        status=render_status(
            code=400,
            status="Bad Request",
            colette_code=1004,
            colette_message="Service Input Bad Request",
        )
    )


def colette_service_input_error_1005(what: str) -> APIResponse:
    return APIResponse(
        status=render_status(
            code=500,
            status="Internal Error",
            colette_code=1005,
            colette_message="Service Input Transform Error: " + what,
        )
    )


def colette_service_bad_request_1006(what: str) -> APIResponse:
    return APIResponse(
        status=render_status(
            code=400,
            status="Bad Request",
            colette_code=1006,
            colette_message="Service Bad Request: " + what,
        )
    )


def colette_service_llmlib_error_1007(what: str) -> APIResponse:
    return APIResponse(
        status=render_status(
            code=500,
            status="Internal Error",
            colette_code=1007,
            colette_message="Internal LLMLib Error: " + what,
        )
    )


# Main JSON API
class JSONApi(APIStrategy):
    def __init__(self):
        super().__init__()
        self.indexing_status = {}
        self.indexing_queue = asyncio.Queue()

    def start_indexing_loop(self):
        self._indexing_task = asyncio.create_task(self._process_indexing_queue())

    async def _process_indexing_queue(self):
        while True:
            sname, ad = await self.indexing_queue.get()
            self.indexing_status[sname] = "running"
            try:
                self.logger_api.debug(f"launching indexing {ad}")
                await asyncio.to_thread(self.index, sname, ad)
                self.indexing_status[sname] = "finished"
            except Exception as e:
                self.indexing_status[sname] = "error"
                self.logger_api.error(f"Indexing error for {sname}: {e}")
                self.logger_api.error(traceback.format_exc())
            self.indexing_queue.task_done()

    ## resources (info, ...)
    def service_info(self) -> APIResponse:
        ##TODO: more service details (e.g. loaded models, ...)
        all_services = self.list_services()
        response = colette_ok_200()
        response.info = InfoObj(services=all_services)
        return response

    def service_create(self, sname: str, ad: APIData) -> APIResponse:
        self.logger_api.info("creating service with name: " + sname)

        try:
            if sname in self.services:
                self.logger_api.error(f"A service with name {sname} already exists")
                raise ServiceBadParamException()

            params = ad.parameters

            # model lib
            if params.llm:
                model_lib = params.llm.inference.lib
            else:
                model_lib = None

            # instantiate the service
            # print(params.input.lib)
            if params.input.lib == "langchain":
                from .backends.langchain.langchaininputconn import LangChainInputConn
                from .backends.langchain.langchainlib import LangChainLib
                from .backends.langchain.langchainmodel import LangChainModel
                from .backends.langchain.langchainoutputconn import LangChainOutputConn

                CLLMService = createLLMService(LangChainLib)
                llmservice = CLLMService(
                    sname,
                    ad.app.verbose,
                    LangChainInputConn(),
                    LangChainOutputConn(),
                    LangChainModel() if model_lib else None,
                )
            elif params.input.lib == "hf":
                from .backends.hf.hfinputconn import HFInputConn
                from .backends.hf.hflib import HFLib
                from .backends.hf.hfmodel import HFModel
                from .backends.hf.hfoutputconn import HFOutputConn

                CLLMService = createLLMService(HFLib)
                llmservice = CLLMService(
                    sname,
                    ad.app.verbose,
                    HFInputConn(),
                    HFOutputConn(),
                    HFModel() if model_lib else None,
                )
            elif params.input.lib == "diffusr":
                from .backends.diffusr.diffusrlib import DiffusrLib
                from .backends.diffusr.diffusrmodel import DiffusrModel

                CLLMService = createLLMService(DiffusrLib)
                llmservice = CLLMService(
                    sname,
                    ad.app.verbose,
                    # DiffusrInputConn(),
                    None,
                    None,
                    DiffusrModel() if model_lib else None,
                )
            else:
                return colette_unknown_library_1000(params.input.lib)

            self.logger_api.info("service using input lib: " + params.input.lib)
            self.add_service(llmservice, sname, ad)

        except InputConnectorBadParamException:
            return colette_service_input_bad_request_1004()
        except LLMLibBadParamException as e:
            return colette_service_bad_request_1006(repr(e))
        except LLMLibInternalException as e:
            return colette_service_llmlib_error_1007(repr(e))
        except ServiceBadParamException:
            return colette_bad_request_400(f"a service with name {sname} already exists")
        except Exception as e:
            self.logger_api.error("Exception: " + str(e))
            self.logger_api.error(traceback.format_exc())

        return APIResponse(service_name=sname)

    async def service_delete(self, sname: str) -> APIResponse:
        self.logger_api.info("deleting service with name: " + sname)
        ret = await asyncio.to_thread(self.remove_service, sname)
        if ret:
            return colette_ok_200()
        else:
            return colette_not_found_404()

    def service_predict(self, sname: str, ad: APIData) -> APIResponse:
        self.logger_api.info("predicting service with name: " + sname)

        try:
            # call on services predict
            response = self.predict(sname, ad)
            response.status = colette_ok_200().status
        except ServiceBadParamException:
            return colette_not_found_404()
        except InputConnectorBadParamException:
            return colette_service_input_bad_request_1004()
        except InputConnectorInternalException as e:
            return colette_service_input_error_1005(repr(e))
        except LLMModelBadParamException as e:
            return colette_unknown_library_1000(repr(e))
        except LLMModelInternalException as e:
            return colette_service_llmlib_error_1007(repr(e))
        except LLMLibBadParamException as e:
            return colette_service_bad_request_1006(repr(e))
        except LLMLibInternalException as e:
            return colette_service_llmlib_error_1007(repr(e))
        except Exception as e:
            self.logger_api.error("Exception: " + str(e))
            self.logger_api.error(traceback.format_exc())

        return response

    def service_streaming(self, sname: str, ad: APIData):
        self.logger_api.info("streaming service with name: " + sname)

        try:
            streamer = self.streaming(sname, ad)
            return streamer
        except Exception as e:
            self.logger_api.error("Exception: " + str(e))
            self.logger_api.error(traceback.format_exc())

    async def service_upload(self, sname: str, ad: APIData, files=None) -> APIResponse:
        status = self.indexing_status.get(sname, "finished")
        if status == "running":
            self.logger_api.error(f"Service {sname} already indexing")
            return colette_service_llmlib_error_1007(repr(f"Service {sname} already indexing"))

        if files is not None:
            app_dir = self.get_service(sname).app_repository
            uploads_dir = app_dir / "uploads"
            os.makedirs(uploads_dir, exist_ok=True)
            for file in files:
                self.logger_api.info(f"Uploading file: {file.filename} in {uploads_dir}")
                with open(uploads_dir / file.filename, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

            ad.parameters.input.data = [DirectoryPath(uploads_dir)]
            self.logger_api.info(f"{len(files)} file(s) uploaded")
        else:
            self.logger_api.info("No file uploaded")

        self.logger_api.info("uploading and indexing new docs with service name: " + sname)
        try:
            self.indexing_status[sname] = "queued"
            await self.indexing_queue.put((sname, ad))
        except Exception as e:
            self.logger_api.error("Indexing enqueue error: " + str(e))
            return colette_service_llmlib_error_1007(repr(e))
        return APIResponse(message=f"Uploading and Indexing for {sname} started.")

    def service_index(self, sname: str, ad: APIData) -> APIResponse:
        return self.index(sname, ad)

    async def service_index_async(self, sname: str, ad: APIData) -> APIResponse:
        status = self.indexing_status.get(sname, "finished")
        if status == "running":
            self.logger_api.error(f"Service {sname} already indexing")
            return colette_service_llmlib_error_1007(repr("Service {sname} already indexing"))

        self.logger_api.info("indexing new docs with service name: " + sname)
        try:
            self.indexing_status[sname] = "queued"
            await self.indexing_queue.put((sname, ad))
        except Exception as e:
            self.logger_api.error("Indexing enqueue error: " + str(e))
            return colette_service_llmlib_error_1007(repr(e))
        return APIResponse(message=f"Indexing for {sname} started.")

    async def service_index_status(self, sname: str) -> APIResponse:
        self.logger_api.error("getting indexing status for service name: " + sname)
        self.logger_api.error(f"indexing_status: {self.indexing_status}")
        status = self.indexing_status.get(sname, "finished")
        self.logger_api.error("service name [" + sname + "] is: " + status)
        return APIResponse(message=f"Indexing for {sname}: {status}.")
