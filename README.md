
# Telegram Downloader Bot

A Telegram bot for downloading media from platforms like YouTube, Instagram, TikTok, Twitter/X, Reddit, and more, . This bot allows users to download videos, audio, and other media by sending URLs in private chats or enabled group chats. It supports multiple quality options, large file handling, and group admin controls.

## Features

- **Media Downloads**: Download videos and audio from various platforms using yt-dlp.
- **Quality Options**: Choose from best quality (`/best`), medium quality (`/dl`), or audio-only (`/audio`) downloads.
- **Group Support**: Auto-downloads links in enabled groups (admin-controlled with `/enablebot` and `/disablebot`).
- **Large File Handling**: Supports splitting large media files or providing external links when files exceed Telegram's limits.
- **Rate Limiting**: Configurable per-user rate limits to prevent abuse.
- **Admin Commands**: Stats (`/stats`) and cache flush (`/flushcache`) for bot administrators.
- **Health Checks**: HTTP endpoint for monitoring bot and database health.
- **Authentication**: Optional authentication system for restricted access (`/auth`).
- **Redis & PostgreSQL**: Uses Redis for caching and PostgreSQL for persistent storage.

## Prerequisites

- Python 3.9+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for media downloading
- PostgreSQL database
- Redis server
- Telegram Bot Token (obtain from [BotFather](https://t.me/BotFather))
- Optional: Sentry for error tracking

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AnimeshKumarRai/telegram_downloader_bot.git
   cd telegram_downloader_bot
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   Create a `.env` file in the project root with the following:
   ```bash
   TELEGRAM_TOKEN=your_telegram_bot_token
   POSTGRES_DSN=postgresql://user:password@localhost:5432/dbname
   REDIS_URL=redis://localhost:6379/0
   DOWNLOAD_DIR=./downloads
   ADMINS=123456789,987654321
   ADMIN_PASS=your_admin_password
   ```

   See `config.py` for all available configuration options.

4. **Set Up yt-dlp**
   Ensure `yt-dlp` is installed and accessible:
   ```bash
   pip install yt-dlp
   ```

5. **Run Database Migrations**
   Ensure your PostgreSQL database is running and apply migrations (if applicable):
   ```bash
   # Example: If using a migration tool like Alembic
   alembic upgrade head
   ```

6. **Start the Bot**
   ```bash
   python app.py
   ```

7. **Docker Setup**
   ```bash
   docker compose up --build
   ```
   Some Docker you can need incase ((Use with Caution)):

   i. Rebuilds without cache, ignoring previous build data and Starts services, may override
   ```bash
   docker compose build --no-cache  
   docker compose up
   ```
   ii. Stops and removes containers, networks, and volumes. Harm: Permanently deletes data in volumes.
   ```bash
   docker compose down -v 
   ```
   iii. Removes all unused containers, images, networks, and volumes. Harm: Irreversible data loss, including unused but needed resources.
   ```bash
   docker system prune -af --volumes
   ```
   iv. Initializes a Docker setup, potentially overwriting existing configs.
   ```bash
   docker init 
   ```


## Usage

1. **Start the Bot**
   - In private chats: Use `/start` to initialize the bot.
   - In groups: Admins can enable the bot with `/enablebot` to allow auto-downloads.

2. **Download Media**
   - Send a URL (e.g., YouTube, Instagram) in a private chat or enabled group.
   - Use commands for specific formats:
     - `/dl <url>`: Download in medium quality.
     - `/best <url>`: Download in best quality.
     - `/audio <url>`: Download audio only.

3. **Group Management**
   - Admins can enable/disable the bot in groups:
     - `/enablebot`: Enable auto-downloads in the group.
     - `/disablebot`: Disable auto-downloads in the group.

4. **Admin Commands**
   - `/stats`: View bot usage statistics (admin-only).
   - `/flushcache`: Clear the Redis cache (admin-only).

5. **Authentication**
   - Use `/auth <password>` to authenticate users (if enabled).

## Configuration

The bot is configured via environment variables in the `.env` file. Key options include:

- `TELEGRAM_TOKEN`: Your Telegram bot token.
- `POSTGRES_DSN`: PostgreSQL connection string.
- `REDIS_URL`: Redis connection string.
- `DOWNLOAD_DIR`: Directory for temporary downloads (default: `./downloads`).
- `MAX_FILESIZE_MB`: Maximum file size for downloads (default: 1900 MB).
- `ADMINS`: Comma-separated Telegram user IDs for admins.
- `ADMIN_PASS`: Password for admin panel access.

See `config.py` for a full list of configuration options and defaults.

## Project Structure

```bash
telegram_downloader_bot/
├── app.py              # Main bot application
├── config.py           # Configuration management
├── database.py         # Database initialization and health checks
├── redis_client.py     # Redis client utilities
├── logger.py           # Logging setup
├── start.py            # /start command handler
├── help.py             # /help command handler
├── download.py         # Download-related command handlers
├── downloader.py       # Download Logic
├── admin.py            # Admin command handlers
├── group.py            # Group-related handlers
├── cleanup.py          # Cleanup loop for temporary files
├── .env                # Environment variables (not tracked)
├── requirements.txt    # Python dependencies
└── more essential files.....
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for media downloading.
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) for Telegram API integration.
- [Pydantic](https://github.com/pydantic/pydantic) for configuration validation.

## Support

For issues or questions, open an issue on GitHub or contact the bot admin via Telegram. More features soon...
