{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

   {% endif %}
.. py:class:: {% if is_own_page %}{{ obj.id }}{% else %}{{ obj.name }}{% endif %}


   {% if obj.docstring %}
   {{ obj.docstring|indent(3) }}
   {% endif %}

   :TYPE ALIAS:
{{ obj.value|format_alias() }}

{% endif %}


