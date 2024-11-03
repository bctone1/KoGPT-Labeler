from flask import Flask
from pathlib import Path
from app.config import config
from app.utils.logging_config import setup_logging, get_logger


def create_app(config_object=config):
    """Flask 애플리케이션 팩토리 함수"""
    # Flask 앱 생성
    app = Flask(__name__,
                template_folder=str(config_object.BASE_DIR / 'templates'),
                static_folder=str(config_object.BASE_DIR / 'static'))

    # 설정 적용
    app.config.from_object(config_object)

    # 로깅 설정
    logger = setup_logging(app)
    logger.info('Application initialization started')

    try:
        # 필요한 디렉토리 생성
        for directory in [
            config_object.UPLOAD_FOLDER,
            config_object.GENERATED_FOLDER,
            config_object.FEEDBACK_FOLDER
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f'Directory created/checked: {directory}')

        # Blueprint 등록
        from app.routes import bp as main_bp
        app.register_blueprint(main_bp)
        logger.info('Blueprint registered')

        logger.info('Application initialization completed')
        return app

    except Exception as e:
        logger.error(f'Error during application initialization: {str(e)}')
        raise