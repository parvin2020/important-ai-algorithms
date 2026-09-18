# فراخوانی کتابخانه محیط‌های شبیه‌سازی
import gymnasium as gym

# فراخوانی الگوریتم PPO از Stable-Baselines3
from stable_baselines3 import PPO

# ۱. ساخت محیط فیزیکی HalfCheetah در شبیه‌ساز MuJoCo
# render_mode="human" باعث می‌شود عملکرد ربات به صورت ۳ بعدی نمایش داده شود
env = gym.make("HalfCheetah-v5", render_mode="human")

# ۲. تعریف مدل یادگیری PPO با شبکه‌های عصبی عمیق (MlpPolicy)
# این مدل ورودی‌های پیوسته فیزیکی (زوایا و سرعت مفصل‌ها) را دریافت می‌کند
model = PPO("MlpPolicy", env, verbose=1)

# ۳. آموزش عامل برای ۵۰,۰۰۰ گام زمانی در محیط فیزیکی MuJoCo
model.learn(total_timesteps=5000)

# ۴. ذخیره‌سازی مدل آموزش‌دیده روی حافظه
model.save("ppo_halfcheetah_mujoco")

# ۵. بازنشانی محیط برای شروع یک اپیزود جدید و تست عامل
observation, info = env.reset()

# ۶. حلقه اجرای ۵۰۰ گام حرکتی جهت مشاهده دویدن چیتا
for step in range(500):
    # دریافت بهترین گشتاور/نیرو برای مفصل‌های چیتا توسط مدل
    action, _ = model.predict(observation, deterministic=True)
    
    # اعمال نیروها به مفصل‌های ربات در شبیه‌ساز MuJoCo
    observation, reward, terminated, truncated, info = env.step(action)
    
    # اگر ربات واژگون شد یا زمان تمام شد، محیط بازنشانی می‌شود
    if terminated or truncated:
        observation, info = env.reset()

# ۷. بستن محیط شبیه‌سازی و پنجره ۳ بعدی
env.close()