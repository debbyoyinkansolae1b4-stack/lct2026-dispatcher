"""Дневной запас расходников бригады: уже начатые визиты расходуют свой первоначальный запас."""
from collections import Counter


def equipment_reason(order, engineer):
    used=Counter()
    for request in order:
        for item,quantity in request.equipment.items():
            if not isinstance(quantity,int) or isinstance(quantity,bool) or quantity<0:
                return 'equipment', f'некорректная потребность в оборудовании: {item}'
            used[item]+=quantity
    for item,quantity in used.items():
        if quantity>engineer.equipment.get(item,0):
            return 'equipment', f'не хватает оборудования «{item}»: нужно {quantity}, выдано на день {engineer.equipment.get(item,0)}'
    return None


def request_equipment(bk_type, hd_type, config):
    return dict(config['by_hd_type'].get(hd_type,config['by_bk_type'].get(bk_type,{})))
